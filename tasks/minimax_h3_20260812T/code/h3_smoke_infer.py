#!/usr/bin/env python
"""Load the released MiniMax-H3 checkpoint and generate video + audio jointly.

This is the ground-truth probe of the replication: everything downstream (the
scaled-down model, the training loop, the evaluation) is written against the
architecture this script exercises, so it runs first and its dumped facts are
what the small model is checked against.

Three things are recorded, not just the mp4:

* the **parameter census** of the transformer, broken down the way the model
  card claims it (``~13B in AdaLN branches`` out of 33B), so the claim is
  checked rather than repeated;
* the **packed-sequence layout** of one real request, i.e. how many rows each
  modality contributes and what the `(t, h, w)` rotary grid looks like, which is
  the contract the small model has to reproduce exactly;
* wall-clock per denoising step, which sets the scale for what the small model
  can afford.

Layout note: the transformer is 61.7 GB and the Qwen3-VL conditioner 62.1 GB in
bfloat16, so on a 4x85.5 GB GH200 node the conditioner gets its own device and
the generation half gets another, following the two-card recipe in the diffusers
docs. Nothing is offloaded to host RAM that way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parameter_census(transformer) -> dict:
    """Break the transformer's parameters down by role.

    The model card's headline decomposition is `33B total, ~13B in AdaLN
    branches`. `adaln_proj.linear` is `Linear(time_embed_dim, 6*hidden*3)`: six
    modulation parameters, three modalities (video/text/audio), one projection
    shared by `norm1` and `norm2`. That single matrix is what the 13B is.
    """
    buckets: dict[str, int] = {}
    for name, param in transformer.named_parameters():
        # `token_refiner` must be tested first: its blocks also match `.attn.` / `.ff.`
        # and would otherwise be counted as part of the 50-block main stack.
        if "token_refiner" in name:
            key = "token_refiner"
        elif "adaln_proj" in name:
            key = "adaln_branches"
        elif ".attn." in name:
            key = "attention"
        elif ".ff." in name:
            key = "feedforward"
        elif name.startswith(("proj_in", "audio_proj_in", "context_embedder")):
            key = "input_projections"
        elif name.startswith(("proj_out", "audio_proj_out", "norm_out")):
            key = "output_heads"
        elif "time_embedder" in name:
            key = "time_embedder"
        else:
            key = "norms_and_other"
        buckets[key] = buckets.get(key, 0) + param.numel()
    buckets["TOTAL"] = sum(v for k, v in buckets.items() if k != "TOTAL")
    return buckets


LAYOUT_FIELDS = ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")


def dump_layout(results: dict, out_path: Path) -> dict:
    """Record the packed-sequence layout the pipeline built for this request.

    The layout lives in the *generation* half's state, not the conditioner's, and a
    call with an explicit `output=` returns only what it names, so the fields are
    requested alongside the media rather than read off the conditioner afterwards.
    This is the contract H3-nano's `build_layout` has to reproduce, so it is worth
    recording from a real request rather than assumed.
    """
    facts: dict = {}
    for field in LAYOUT_FIELDS:
        value = results.get(field)
        if isinstance(value, torch.Tensor):
            facts[field] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if field == "token_tags":
                # 0 = video, 1 = text, 2 = audio -- MINIMAX_H3_MODALITY_NUM = 3.
                counts = torch.bincount(value.flatten().cpu(), minlength=3).tolist()
                facts[field]["rows_per_modality"] = {"video": counts[0], "text": counts[1], "audio": counts[2]}
                facts[field]["sequence_length"] = int(value.numel())
            if field == "position_ids":
                grid = value.float().cpu()
                facts[field]["min"] = grid.amin(dim=0).tolist()
                facts[field]["max"] = grid.amax(dim=0).tolist()
    out_path.write_text(json.dumps(facts, indent=2))
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Local path of the H3 checkpoint (diffusers layout)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num-frames", type=int, default=124, help="Snapped up to the next 17*n+5")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default=(
            "integrated_multimodal_description: [Shot 1] Cinematic medium shot, static camera. "
            "A red fox trots left to right through a snowy pine forest in soft overcast daylight, "
            "its breath visible, paws breaking through a thin crust of snow with each step. "
            "Fine snow drifts down through the frame throughout the shot.\n"
            "overall_soundscape: Crisp rhythmic crunching of paws compressing dry snow, one crunch per step, "
            "over a quiet forest room tone with a faint high wind through pine needles.\n"
            "non_diegetic_music: None."
        ),
        help="H3 expects a Context-IR style structured prompt, not a bare caption.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import ModularPipeline
    from diffusers.utils.export_utils import encode_video

    n_gpu = torch.cuda.device_count()
    print(f"[smoke] visible GPUs: {n_gpu}", flush=True)
    for i in range(n_gpu):
        free, total = torch.cuda.mem_get_info(i)
        print(f"[smoke]   cuda:{i} {torch.cuda.get_device_name(i)}  free {free/2**30:.1f} / {total/2**30:.1f} GiB",
              flush=True)
    if n_gpu < 2:
        raise SystemExit("[smoke] need >= 2 GPUs for the split-pipeline recipe")

    # The conditioner half and the generation half each get their own device: with
    # ~62 GB per half and 85.5 GB per GH200 there is room for both to stay resident,
    # which avoids paying host-RAM round trips on every denoising step.
    print("[smoke] building split pipeline (conditioner on cuda:1, denoiser on cuda:0)", flush=True)
    t0 = time.time()
    workflow = ModularPipeline.from_pretrained(args.ckpt).blocks.get_workflow("t2va")

    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(args.ckpt)
    conditioner.load_components(dtype=torch.bfloat16)
    conditioner.text_encoder.to("cuda:1")

    rest = workflow.init_pipeline(args.ckpt)
    rest.load_components(dtype=torch.bfloat16)
    for name in ("transformer", "vae", "audio_vae"):
        component = getattr(rest, name, None)
        if component is not None:
            component.to("cuda:0")
    print(f"[smoke] components loaded in {time.time()-t0:.0f}s", flush=True)

    # Hopper: FlashAttention-3 kernels are fetched from the Hub and are ~3x faster.
    try:
        rest.transformer.set_attention_backend("_flash_3_hub")
        print("[smoke] attention backend: _flash_3_hub", flush=True)
    except Exception as exc:  # a missing kernel must not fail the probe
        print(f"[smoke] flash_3 unavailable ({exc}); using the default backend", flush=True)

    census = parameter_census(rest.transformer)
    print("[smoke] parameter census (billions):", flush=True)
    for key, value in sorted(census.items(), key=lambda kv: -kv[1]):
        print(f"[smoke]   {key:22s} {value/1e9:8.3f} B", flush=True)
    (out_dir / "param_census.json").write_text(json.dumps(census, indent=2))

    print(f"[smoke] generating {args.width}x{args.height}, {args.num_frames} frames, {args.steps} steps", flush=True)
    t0 = time.time()
    state = conditioner(prompt=args.prompt)
    t_cond = time.time() - t0

    call_kwargs = dict(
        state=state,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
    )
    media = ["videos", "audio", "sampling_rate"]
    t0 = time.time()
    try:
        results = rest(**call_kwargs, output=media + list(LAYOUT_FIELDS))
    except Exception as exc:
        # An older block set may not expose the layout by name; the media still is
        # the point of the run, so fall back rather than lose the generation.
        print(f"[smoke] layout outputs unavailable ({type(exc).__name__}: {exc}); "
              f"requesting media only", flush=True)
        results = rest(**call_kwargs, output=media)
    t_gen = time.time() - t0

    video = results["videos"][0]
    audio = results["audio"][0]
    rate = results["sampling_rate"]
    n_frames = len(video) if hasattr(video, "__len__") else "?"
    print(f"[smoke] conditioning {t_cond:.1f}s | denoising {t_gen:.1f}s "
          f"({t_gen/max(args.steps-1,1):.2f}s/step)", flush=True)
    print(f"[smoke] video frames={n_frames}  audio shape={tuple(audio.shape)} @ {rate} Hz", flush=True)

    mp4 = out_dir / "h3_t2va_smoke.mp4"
    encode_video(video, fps=24, output_path=str(mp4), audio=audio, audio_sample_rate=rate)
    print(f"[smoke] wrote {mp4} ({mp4.stat().st_size/1e6:.1f} MB)", flush=True)

    layout = dump_layout(results, out_dir / "packed_layout.json")
    print(f"[smoke] packed layout: {json.dumps(layout.get('token_tags', {}))}", flush=True)

    (out_dir / "timings.json").write_text(json.dumps({
        "conditioning_s": t_cond, "denoise_s": t_gen, "steps": args.steps,
        "s_per_step": t_gen / max(args.steps - 1, 1),
        "height": args.height, "width": args.width, "num_frames": args.num_frames,
        "audio_samples": int(audio.shape[-1]), "sampling_rate": int(rate),
    }, indent=2))
    print("[smoke] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
