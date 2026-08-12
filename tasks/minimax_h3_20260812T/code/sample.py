#!/usr/bin/env python
"""Sample video + audio from an H3-nano checkpoint and decode it with H3's own VAEs.

The denoising loop is the released one: two `MiniMaxH3Scheduler` instances stepped
inside a single transformer call per iteration, `shift = 12` for the video rows and
`shift = 3` for the audio rows. Reusing the scheduler class rather than reimplementing
Euler means the sampled trajectory is the reference trajectory, including its two
deliberate quirks -- the data-ward velocity sign (`x0 = x_t + sigma * v`) and the
recovery of sigma from the conditioning timestep rather than from the grid.

Guidance is a flag rather than a constant because it is what the distillation stage
removes: a pretrained or SFT checkpoint needs `--guidance-scale w` and pays two
forward passes per step, while the distilled checkpoint is called with
`--guidance-scale 0` and pays one, which is the released model's contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3nano as H  # noqa: E402


@torch.no_grad()
def sample(model, corpus_text, label_indices, layout, latent_shape, audio_shape,
           steps: int, guidance: float, null_index: int, device, seed: int = 0):
    """Run the two-schedule denoising loop and return normalized latents.

    Returns `(video_latents, audio_latents)` still in the VAE's normalized space; the
    caller denormalizes and decodes.
    """
    from diffusers import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler(shift=H.VIDEO_FLOW_SHIFT)
    audio_scheduler = MiniMaxH3Scheduler(shift=H.AUDIO_FLOW_SHIFT)
    scheduler.set_timesteps(steps, device=device)
    audio_scheduler.set_timesteps(steps, device=device)

    batch = len(label_indices)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Draw order matters for reproducibility against the reference: conditioning noise
    # first (none here), then video, then audio.
    video = torch.randn(latent_shape, generator=generator).to(device)
    audio = torch.randn(audio_shape, generator=generator).to(device)

    video_rows = H.patchify_video_latents(video)
    audio_rows = H.pack_audio_channel_major(audio)

    text = corpus_text[label_indices].to(device, torch.float32)
    null_text = corpus_text[torch.full((batch,), null_index)].to(device, torch.float32)

    n_cond = layout.num_condition_video_rows
    for i, t in enumerate(scheduler.timesteps):
        at = audio_scheduler.timesteps[i]
        timestep, timestep_indices = H.build_row_timesteps(layout, float(t), float(at))
        kwargs = dict(
            hidden_states=video_rows, audio_hidden_states=audio_rows,
            encoder_hidden_states=text, timestep=timestep.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=layout.token_tags, position_ids=layout.position_ids,
            video_indices=layout.video_indices, audio_indices=layout.audio_indices,
            text_indices=layout.text_indices, return_dict=False,
        )
        v_pred, a_pred = model(**kwargs)
        if guidance > 0:
            kwargs["encoder_hidden_states"] = null_text
            v_unc, a_unc = model(**kwargs)
            v_pred = v_unc + guidance * (v_pred - v_unc)
            a_pred = a_unc + guidance * (a_pred - a_unc)

        video_rows[:, n_cond:] = scheduler.step(
            v_pred[:, n_cond:].float(), t, video_rows[:, n_cond:], return_dict=False)[0]
        audio_rows = audio_scheduler.step(a_pred.float(), at, audio_rows, return_dict=False)[0]

    video = H.unpatchify_video_latents(video_rows[:, n_cond:], layout.num_latent_frames,
                                       layout.latent_height, layout.latent_width)
    audio = H.unpack_audio_channel_major(audio_rows)
    return video, audio


@torch.no_grad()
def decode(video_latents, audio_latents, ckpt: Path, device):
    """Denormalize and decode with the frozen H3 VAEs, returning pixels and waveform."""
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3.from_pretrained(ckpt / "vae", dtype=torch.bfloat16).to(device).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(ckpt / "audio_vae",
                                                            dtype=torch.float32).to(device).eval()

    lat_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    a_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, 1, -1)
    a_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, 1, -1)

    latents = video_latents.to(device) * lat_std + lat_mean
    pixels = vae.decode(latents.to(torch.bfloat16), return_dict=False)[0].float()
    # Undo the ImageNet normalization the encoder applied over a [0, 1] base.
    mean = torch.tensor(H.PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(H.PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    pixels = ((pixels * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)

    waves = []
    for item in audio_latents:                                   # (2, 32, L) per item
        # Denormalization is per latent channel, so it happens on the (2, L, 32) view
        # the statistics were shaped for, then the channel axis goes back in front.
        lat = item.to(device).transpose(1, 2)                     # (2, L, 32)
        lat = ((lat * a_std) + a_mean).transpose(1, 2)            # (2, 32, L)
        wave = audio_vae.decode(lat, return_dict=False)[0]        # (2, 1, samples)
        waves.append(wave.squeeze(1).float().cpu())
    return pixels.cpu(), torch.stack(waves)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir holding latest_checkpoint.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=None, help="Comma-separated class names; default = first 4")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="0 for a distilled checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root, out_dir = Path(args.root), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    latents_dir = root / "data" / "latents"
    manifest = json.loads((latents_dir / "manifest.json").read_text())
    bank = torch.load(latents_dir / "text_bank.pt", map_location="cpu", weights_only=False)
    labels = bank["labels"]

    info = json.loads((Path(args.checkpoint) / "latest_checkpoint.json").read_text())
    blob = torch.load(info["path"], map_location="cpu", weights_only=False)
    config = blob["config"]["arch"]
    model = H.build_transformer(config).to(device).eval()
    model.load_state_dict(blob["model"])
    print(f"[sample] {info['path']} step {blob['step']} | "
          f"{H.parameter_census(model)['TOTAL']/1e6:.1f} M params", flush=True)

    wanted = [l.strip() for l in args.labels.split(",")] if args.labels else labels[:4]
    label_indices = torch.tensor([labels.index(w) for w in wanted])

    latent_frames = manifest["latent_frames"]
    audio_len = manifest["audio_latents"]
    latent_hw = manifest["size"] // H.VAE_SPATIAL_COMPRESSION
    layout = H.build_layout(bank["max_tokens"], latent_frames, latent_hw, latent_hw, audio_len).to(device)

    t0 = time.time()
    video_lat, audio_lat = sample(
        model, bank["embeds"], label_indices, layout,
        (len(wanted), 24, latent_frames, latent_hw, latent_hw),
        (len(wanted), H.AUDIO_CHANNELS, 32, audio_len),
        args.steps, args.guidance_scale, bank["null_index"], device, args.seed)
    passes = args.steps - 1 if args.guidance_scale <= 0 else 2 * (args.steps - 1)
    print(f"[sample] {time.time()-t0:.1f}s for {args.steps-1} steps "
          f"({passes} forward passes, guidance={args.guidance_scale})", flush=True)

    pixels, waves = decode(video_lat, audio_lat, root / "ckpt" / "h3", device)
    print(f"[sample] decoded video {tuple(pixels.shape)} audio {tuple(waves.shape)}", flush=True)

    from diffusers.utils.export_utils import export_to_video
    try:
        from diffusers.utils.export_utils import encode_video
    except ImportError:
        encode_video = None

    for i, name in enumerate(wanted):
        frames = [np.asarray(f) for f in pixels[i].permute(1, 2, 3, 0).numpy()]
        slug = name.replace(" ", "_").replace("/", "-")
        path = out_dir / f"{slug}_s{args.seed}.mp4"
        if encode_video is not None:
            encode_video(frames, fps=int(H.FPS), output_path=str(path),
                         audio=waves[i], audio_sample_rate=H.AUDIO_SAMPLING_RATE)
        else:
            export_to_video(frames, str(path), fps=int(H.FPS))
        print(f"[sample] wrote {path}", flush=True)

    torch.save({"video_latents": video_lat.cpu(), "audio_latents": audio_lat.cpu(),
                "labels": wanted, "step": blob["step"], "guidance": args.guidance_scale},
               out_dir / "samples.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
