#!/usr/bin/env python
"""Measure H3-nano against the statements MiniMax actually made about H3.

Each subcommand turns one documented claim into a number. Nothing here scores
"quality" in the abstract: with no technical report there is no reported benchmark
to match, so what is checkable is whether the *mechanisms* the release documents
are real and whether the small model reproduces them.

    roundtrip   E6  VAE encode->decode fidelity. A gate, not a result: if this is
                    bad the preprocessing is wrong and no training number means
                    anything.
    heldout     E3  Held-out flow-matching loss at pinned timesteps, so two
                    E7      checkpoints are scored on identical noise levels.
    avsync      E1  Audio-envelope vs frame-difference cross-correlation, against
                    a mismatched-pair control.
    anchors     E4  "The released model was trained with its anchors very slightly
                    noised, so conditioning on exactly t = 1.0 is off-distribution."
    speed       E2  "Every step runs exactly one forward pass" -- forward-pass
                    count and wall-clock, guided vs distilled.
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


def si_sdr(target: torch.Tensor, estimate: torch.Tensor) -> float:
    """Scale-invariant SDR in dB. Scale-invariant, not shift-invariant: a few samples
    of misalignment destroy it, which is why the time axis is truncated before any
    flattening happens."""
    alpha = (target @ estimate) / (target @ target + 1e-9)
    noise = estimate - alpha * target
    return float(10 * torch.log10((alpha * target).pow(2).sum() / (noise.pow(2).sum() + 1e-9)))


def load_model(checkpoint: Path, device):
    info = json.loads((checkpoint / "latest_checkpoint.json").read_text())
    blob = torch.load(info["path"], map_location="cpu", weights_only=False)
    model = H.build_transformer(blob["config"]["arch"]).to(device).eval()
    model.load_state_dict(blob["model"])
    return model, blob["step"], blob["config"]


# ---------------------------------------------------------------- E6
@torch.no_grad()
def cmd_roundtrip(args) -> dict:
    """Encode and decode real clips through the frozen VAEs and score the round trip.

    This runs *before* any training conclusion is drawn. A latent corpus built with
    the wrong pixel convention (H3 uses ImageNet normalization over `[0, 1]`, not
    the usual `[-1, 1]`) or the wrong latent statistics still trains to a plausible
    loss curve and produces garbage, and this is the only cheap way to catch it.
    """
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    root = Path(args.root)
    device = torch.device(args.device)
    ckpt = root / "ckpt" / "h3"
    # float32, not bfloat16. The reference's `encode_vae_condition` feeds
    # `pixels.to(torch.float32)` and this VAE keeps some biases in fp32 (the same
    # mixed-precision design as the transformer's `_keep_in_fp32_modules`), so a
    # bf16 input meets an fp32 bias:
    #   RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
    # At 20.8 GB in fp32 against 94 GB free, there is nothing to buy by narrowing it.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(ckpt / "vae", dtype=torch.float32).to(device).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(ckpt / "audio_vae",
                                                            dtype=torch.float32).to(device).eval()

    import preprocess_vggsound as P
    import tarfile
    from concurrent.futures import ProcessPoolExecutor

    tarball = root / "data" / "vggsound" / f"vggsound_{args.shard:02d}.tar.gz"
    num_frames = H.snap_num_frames(args.num_frames)
    payloads = []
    with tarfile.open(tarball, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".mp4"):
                handle = tar.extractfile(member)
                if handle is not None:
                    payloads.append((Path(member.name).stem + "|x", handle.read()))
            if len(payloads) >= args.n:
                break

    with ProcessPoolExecutor(max_workers=8) as pool:
        clips = [c for c in pool.map(P._decode_partial(num_frames, args.size), payloads) if c]
    print(f"[roundtrip] decoded {len(clips)}/{len(payloads)} clips", flush=True)

    pixel_mean = torch.tensor(H.PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(H.PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

    # Reconstructions are kept so a mismatched control can be scored afterwards.
    # SI-SDR on arbitrary YouTube audio is strongly content-dependent -- quiet clips
    # score badly however good the codec is -- so an absolute threshold picked in
    # advance says little. What does say something is whether a reconstruction
    # resembles *its own* source more than someone else's.
    psnrs, sisdrs, recon_audio, orig_audio = [], [], [], []
    for _name, video_np, audio_np in clips:
        original = torch.from_numpy(video_np).to(device).permute(3, 0, 1, 2)[None].float()
        pixels = (original.div(255.0) - pixel_mean) / pixel_std
        latents = vae.encode(pixels, return_dict=False)[0].sample(
            generator=torch.Generator(device=device).manual_seed(42))
        recon = vae.decode(latents, return_dict=False)[0].float()
        recon = ((recon * pixel_std + pixel_mean).clamp(0, 1) * 255)
        mse = (recon - original).pow(2).mean()
        psnrs.append(float(10 * torch.log10(255.0 ** 2 / mse)))

        wave = torch.from_numpy(audio_np).to(device)[:, None]
        a_lat = audio_vae.encode(wave, return_dict=False)[0].mode()
        a_rec = audio_vae.decode(a_lat, return_dict=False)[0]
        # Truncate on the *time* axis before flattening. The VAE returns whole latent
        # frames -- 122 x 800 = 97,600 samples for a 97,333-sample input -- so
        # `a_rec.flatten()[:wave.numel()]` keeps channel 0 aligned and shifts channel 1
        # by the 267-sample difference. SI-SDR is extremely sensitive to alignment, and
        # that shift alone drove it to -6.88 dB: the number was measuring the flatten
        # order, not the reconstruction.
        n = min(wave.shape[-1], a_rec.shape[-1])
        target, estimate = wave[..., :n].flatten(), a_rec[..., :n].flatten()
        sisdrs.append(si_sdr(target, estimate))
        recon_audio.append(a_rec[..., :n].flatten().cpu())
        orig_audio.append(wave[..., :n].flatten().cpu())

    # The control: each reconstruction against the *next* clip's source.
    control = []
    for i in range(len(recon_audio)):
        j = (i + 1) % len(recon_audio)
        m = min(recon_audio[i].numel(), orig_audio[j].numel())
        control.append(si_sdr(orig_audio[j][:m], recon_audio[i][:m]))

    result = {
        "clips": len(clips), "num_frames": num_frames, "size": args.size,
        "video_psnr_db": {"mean": float(np.mean(psnrs)), "min": float(np.min(psnrs)),
                          "max": float(np.max(psnrs))},
        "audio_si_sdr_db": {"mean": float(np.mean(sisdrs)), "min": float(np.min(sisdrs)),
                            "max": float(np.max(sisdrs))},
    }
    result["audio_si_sdr_control_db"] = {"mean": float(np.mean(control)),
                                        "min": float(np.min(control)),
                                        "max": float(np.max(control))}
    result["audio_margin_db"] = result["audio_si_sdr_db"]["mean"] - result["audio_si_sdr_control_db"]["mean"]
    print(f"[roundtrip] video PSNR {result['video_psnr_db']['mean']:.2f} dB | "
          f"audio SI-SDR {result['audio_si_sdr_db']['mean']:.2f} dB "
          f"(mismatched control {result['audio_si_sdr_control_db']['mean']:.2f} dB, "
          f"margin {result['audio_margin_db']:+.2f} dB)", flush=True)
    return result


# ---------------------------------------------------------------- E3 / E7
@torch.no_grad()
def cmd_heldout(args) -> dict:
    """Held-out flow loss on a grid of *pinned* timesteps.

    Timesteps are pinned rather than drawn because the loss is a strong function of
    `t`: a fresh draw per checkpoint would put more variance into the draw than
    into the difference between the models being compared. The grid is the same
    shifted grid the sampler walks, so the numbers are weighted toward the noise
    levels that actually matter at inference.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import LatentCorpus

    root = Path(args.root)
    device = torch.device(args.device)
    corpus = LatentCorpus(root / "data" / "latents")
    model, step, config = load_model(Path(args.checkpoint), device)

    latent_frames, latent_h, latent_w = corpus.video.shape[2], corpus.video.shape[3], corpus.video.shape[4]
    layout = H.build_layout(corpus.text.shape[1], latent_frames, latent_h, latent_w,
                            corpus.audio.shape[3]).to(device)

    # Hold out the tail of the corpus. Preprocessing writes clips in tarball order,
    # which is YouTube-id order, so a tail split is not class-stratified; it is a
    # split by *clip*, which is what matters for a generative loss.
    n_val = min(args.n, len(corpus) // 5)
    index = torch.arange(len(corpus) - n_val, len(corpus))

    sigma_grid = torch.linspace(1.0, 0.0, args.grid + 1)[:-1]
    rows = []
    for sigma in sigma_grid:
        v_t = float(1.0 - H.apply_shift(sigma, H.VIDEO_FLOW_SHIFT))
        a_t = float(1.0 - H.apply_shift(sigma, H.AUDIO_FLOW_SHIFT))
        totals = {"loss": 0.0, "loss_video": 0.0, "loss_audio": 0.0}
        seen = 0
        for start in range(0, n_val, args.batch_size):
            pick = index[start:start + args.batch_size]
            if len(pick) == 0:
                continue
            video = corpus.video[pick].to(device, torch.float32)
            audio = corpus.audio[pick].to(device, torch.float32)
            text = corpus.text[corpus.label[pick]].to(device, torch.float32)
            generator = torch.Generator().manual_seed(1234 + start)
            batch = H.make_flow_batch(video, audio, text, layout, generator=generator,
                                      fixed_t=(v_t, a_t))
            v_pred, a_pred = model(**batch.transformer_kwargs())
            _loss, logs = H.flow_loss(v_pred, a_pred, batch)
            for key in totals:
                totals[key] += logs[key] * len(pick)
            seen += len(pick)
        rows.append({"sigma": float(sigma), "t_video": v_t, "t_audio": a_t,
                     **{k: v / max(seen, 1) for k, v in totals.items()}})
        print(f"[heldout] sigma {float(sigma):.3f}  loss {rows[-1]['loss']:.4f} "
              f"(v {rows[-1]['loss_video']:.4f} a {rows[-1]['loss_audio']:.4f})", flush=True)

    mean_loss = float(np.mean([r["loss"] for r in rows]))
    print(f"[heldout] step {step}: mean loss over grid = {mean_loss:.4f}", flush=True)
    return {"checkpoint": str(args.checkpoint), "step": step, "n_val": n_val,
            "arch_hidden": config["arch"]["hidden_size"], "arch_layers": config["arch"]["num_layers"],
            "mean_loss": mean_loss, "grid": rows}


# ---------------------------------------------------------------- E1
def envelope(wave: np.ndarray, hop: int) -> np.ndarray:
    """Short-time RMS of a waveform, on a frame grid."""
    mono = wave.mean(axis=0) if wave.ndim > 1 else wave
    n = len(mono) // hop
    return np.sqrt((mono[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)


def motion_energy(frames: np.ndarray) -> np.ndarray:
    """Mean absolute frame-to-frame difference, one value per frame transition."""
    diff = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
    return diff.reshape(len(diff), -1).mean(axis=1)


def sync_lag(audio_env: np.ndarray, motion: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in frames) and peak of the cross-correlation, normalized per lag.

    Each lag is scored as a Pearson correlation over *its own* overlap. Normalizing
    by the full-length norms instead would divide a shrinking numerator (fewer terms
    survive as the lag grows) by a constant denominator, which biases the peak
    toward lag 0 -- and lag 0 is exactly what this measurement is trying to
    establish, so the estimator would be answering its own question.

    **Sign convention**: a *negative* lag means the soundtrack trails the picture,
    i.e. the sound of an impact arrives after the impact is visible. Checked against
    a planted shift: `audio = roll(motion, +4)` (audio delayed by four frames) is
    recovered as lag `-4`. A well-synchronized generation should sit near 0; a
    consistent non-zero lag is a real finding, not noise, and its sign says which
    modality the model is leading with.
    """
    n = min(len(audio_env), len(motion))
    audio_env, motion = audio_env[:n], motion[:n]
    lags = list(range(-max_lag, max_lag + 1))
    scores = []
    for lag in lags:
        if lag < 0:
            left, right = audio_env[-lag:], motion[: n + lag]
        elif lag > 0:
            left, right = audio_env[: n - lag], motion[lag:]
        else:
            left, right = audio_env, motion
        if len(left) < 8:
            scores.append(0.0)
            continue
        a = left - left.mean()
        m = right - right.mean()
        denom = np.sqrt((a ** 2).sum() * (m ** 2).sum())
        scores.append(float((a * m).sum() / denom) if denom > 1e-12 else 0.0)
    best = int(np.argmax(scores))
    return lags[best], scores[best]


def cmd_avsync(args) -> dict:
    """Cross-correlate the soundtrack's envelope against visible motion.

    The control is the point of the measurement: a positive correlation on matched
    pairs means nothing unless mismatched pairs -- this clip's video against that
    clip's audio -- score lower. Video and audio that merely share a loudness
    envelope shape would pass without the control.
    """
    blob = torch.load(Path(args.samples), map_location="cpu", weights_only=False)
    frames_all = blob["frames"] if "frames" in blob else None
    waves_all = blob["waves"] if "waves" in blob else None
    if frames_all is None or waves_all is None:
        raise SystemExit(f"{args.samples} carries no decoded media; run sample.py --save-decoded")

    hop = int(H.AUDIO_SAMPLING_RATE / H.FPS)
    matched, control = [], []
    n = len(frames_all)
    for i in range(n):
        frames = np.asarray(frames_all[i])
        env = envelope(np.asarray(waves_all[i]), hop)
        lag, peak = sync_lag(env, motion_energy(frames), args.max_lag)
        matched.append({"index": i, "lag_frames": lag, "peak": peak})
        j = (i + 1) % n                      # this video against the next clip's audio
        env_other = envelope(np.asarray(waves_all[j]), hop)
        _lag2, peak2 = sync_lag(env_other, motion_energy(frames), args.max_lag)
        control.append(peak2)

    result = {
        "n": n,
        "matched_peak_mean": float(np.mean([m["peak"] for m in matched])),
        "control_peak_mean": float(np.mean(control)),
        "median_lag_frames": float(np.median([m["lag_frames"] for m in matched])),
        "per_clip": matched,
    }
    result["margin"] = result["matched_peak_mean"] - result["control_peak_mean"]
    print(f"[avsync] matched {result['matched_peak_mean']:.4f} vs control "
          f"{result['control_peak_mean']:.4f} (margin {result['margin']:+.4f}), "
          f"median lag {result['median_lag_frames']:+.1f} frames", flush=True)
    return result


# ---------------------------------------------------------------- E4
@torch.no_grad()
def cmd_anchors(args) -> dict:
    """Sweep the anchor timestep of an fl2va checkpoint around 0.999.

    If the released model's `keyframe_noise_aug = 0.999` really is a property of how
    it was *trained*, then a model trained the same way should do worse when the
    anchor is presented at exactly `t = 1.0`, which it never saw. That is a
    falsifiable prediction about the training recipe, and this measures it.
    """
    from train import LatentCorpus, make_condition_rows

    root = Path(args.root)
    device = torch.device(args.device)
    corpus = LatentCorpus(root / "data" / "latents")
    model, step, config = load_model(Path(args.checkpoint), device)
    anchors = tuple(a for a in config["anchors"].split(",") if a)

    latent_frames, latent_h, latent_w = corpus.video.shape[2], corpus.video.shape[3], corpus.video.shape[4]
    layout = H.build_layout(corpus.text.shape[1], latent_frames, latent_h, latent_w,
                            corpus.audio.shape[3], keyframe_anchors=anchors).to(device)

    n_val = min(args.n, len(corpus) // 5)
    index = torch.arange(len(corpus) - n_val, len(corpus))
    rows = []
    for anchor_t in [float(x) for x in args.anchor_grid.split(",")]:
        total, seen = 0.0, 0
        for start in range(0, n_val, args.batch_size):
            pick = index[start:start + args.batch_size]
            if len(pick) == 0:
                continue
            video = corpus.video[pick].to(device, torch.float32)
            audio = corpus.audio[pick].to(device, torch.float32)
            text = corpus.text[corpus.label[pick]].to(device, torch.float32)
            generator = torch.Generator().manual_seed(999 + start)
            condition = make_condition_rows(video, anchors, anchor_t=anchor_t)
            batch = H.make_flow_batch(video, audio, text, layout, generator=generator,
                                      condition_rows=condition, fixed_t=(args.t_video, args.t_audio))
            # The row-timestep table has to state the anchor level actually used, or
            # the model is told 0.999 while being shown something else.
            timestep, timestep_indices = H.build_row_timesteps(
                layout, args.t_video, args.t_audio, condition_video_t=anchor_t)
            batch.timestep, batch.timestep_indices = timestep.to(device), timestep_indices.to(device)
            v_pred, a_pred = model(**batch.transformer_kwargs())
            _loss, logs = H.flow_loss(v_pred, a_pred, batch)
            total += logs["loss_video"] * len(pick)
            seen += len(pick)
        rows.append({"anchor_t": anchor_t, "loss_video": total / max(seen, 1)})
        print(f"[anchors] anchor t={anchor_t:.4f} -> video loss {rows[-1]['loss_video']:.5f}", flush=True)

    best = min(rows, key=lambda r: r["loss_video"])
    at_one = next((r for r in rows if r["anchor_t"] == 1.0), None)
    print(f"[anchors] best anchor t = {best['anchor_t']}", flush=True)
    return {"checkpoint": str(args.checkpoint), "step": step, "grid": rows,
            "best_anchor_t": best["anchor_t"],
            "penalty_at_t1": (at_one["loss_video"] - best["loss_video"]) if at_one else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="Write the result JSON here")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("roundtrip", help="E6: VAE fidelity gate")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--num-frames", type=int, default=73)
    p.add_argument("--size", type=int, default=256)
    p.set_defaults(func=cmd_roundtrip)

    p = sub.add_parser("heldout", help="E3/E7: held-out flow loss at pinned timesteps")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grid", type=int, default=8)
    p.set_defaults(func=cmd_heldout)

    p = sub.add_parser("avsync", help="E1: audio/motion cross-correlation with a control")
    p.add_argument("--samples", required=True)
    p.add_argument("--max-lag", type=int, default=12)
    p.set_defaults(func=cmd_avsync)

    p = sub.add_parser("anchors", help="E4: the keyframe_noise_aug = 0.999 prediction")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--anchor-grid", default="1.0,0.999,0.99,0.95,0.9")
    p.add_argument("--t-video", type=float, default=0.5)
    p.add_argument("--t-audio", type=float, default=0.5)
    p.set_defaults(func=cmd_anchors)

    args = parser.parse_args()
    result = args.func(args)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[eval] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
