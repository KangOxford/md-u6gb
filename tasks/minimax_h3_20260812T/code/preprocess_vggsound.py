#!/usr/bin/env python
"""Turn VGGSound mp4s into the latents H3-nano trains on, using H3's own frozen encoders.

The point of reusing the released VAEs and conditioner rather than training small
ones is that it fixes the comparison: H3-nano then denoises in *the same* latent
space, conditioned on *the same* text features, as the 33B model. Any difference
in what the two learn is the transformer's, not the representation's.

Two phases, because their costs are wildly different.

``--phase text`` builds the text bank. VGGSound's caption is one of 309 class
labels, and the Context-IR template below depends on nothing else, so there are
only 309 distinct prompts in the whole corpus. The 62 GB Qwen3-VL conditioner
therefore runs 309 times, not once per clip, and is then never needed again.

``--phase media`` streams a shard tarball, decodes each clip, and encodes it with
the video and audio VAEs. The tarball is read sequentially and never unpacked:
one large sequential read in, one latent shard out, no small-file metadata churn
on Lustre.

Every step of the encoding recipe is the released pipeline's, transcribed from
``diffusers...modular_pipelines/minimax_h3/encoders.py``:

* pixels are ImageNet-normalized over a ``[0, 1]`` base, not the usual ``[-1, 1]``;
* the video posterior is **sampled** and rounded through float16;
* the audio posterior is **never** sampled, it takes the mode;
* both latents are normalized per channel by the VAE's ``latents_mean/std``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3nano as H  # noqa: E402


# ----------------------------------------------------------------------------
# A miniature Context-IR
# ----------------------------------------------------------------------------
# MiniMax's H3-Context-IR is a hosted, closed system that rewrites a user request
# into a structured document before H3 ever sees it; the released examples run
# 5,650 to 33,323 tokens over the fixed sections below. It is not open, so this is
# a stand-in that keeps the *shape* -- the same three section names, in the same
# order, describing shot, diegetic sound and score separately -- while being
# generated from a class label by template. Keeping the shape matters because the
# section names are what the conditioner sees; keeping the content honest matters
# because a template cannot invent detail VGGSound's one-phrase label does not have.
CONTEXT_IR_TEMPLATE = (
    "integrated_multimodal_description: [Shot 1] A single continuous live-action shot at 24 fps, "
    "handheld framing, natural lighting, no cuts. The shot shows {label}, held in frame for the "
    "whole clip, with the motion that produces the sound clearly visible.\n"
    "overall_soundscape: The diegetic sound of {label}, recorded close and continuous, "
    "synchronized frame-accurately to the visible motion, over a quiet ambient room tone.\n"
    "non_diegetic_music: None."
)


def context_ir_prompt(label: str) -> str:
    return CONTEXT_IR_TEMPLATE.format(label=label.strip())


def pick_device(need_gb: float, requested: str = "cuda:0") -> str:
    """Choose a CUDA device with room, at the moment of use rather than at launch.

    Nothing here can reserve a card. GPU declarations were removed on 2026-08-14
    because a declaration outlives the process that made it, so a card that reads
    idle can be filled by a neighbour during the five to ten minutes it takes to
    load weights -- which is exactly what happened on nid010463, where a neighbour
    took 90.17 GiB between the gtop check and the first `.to(device)`.

    Since a card cannot be held, the answer is to look late and to have somewhere to
    go: pick the emptiest card that clears `need_gb` right before allocating. A
    caller that still OOMs can call this again and get a different one.
    """
    if not requested.startswith("cuda") or not torch.cuda.is_available():
        return requested
    best, best_free = None, 0.0
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        free_gb = free / 2**30
        if free_gb > best_free:
            best, best_free = index, free_gb
    if best is None or best_free < need_gb:
        raise SystemExit(
            f"[env] FATAL: no CUDA device has {need_gb:.1f} GiB free (best is "
            f"{best_free:.1f} GiB on cuda:{best}). Every card in this allocation is "
            f"occupied; wait or use another node."
        )
    chosen = f"cuda:{best}"
    if chosen != requested:
        print(f"[env] {requested} lacks room; using {chosen} with {best_free:.1f} GiB free",
              flush=True)
    else:
        print(f"[env] {chosen} has {best_free:.1f} GiB free", flush=True)
    return chosen


def stable_seed(key: str) -> int:
    """A per-clip seed that is the same in every process and every run.

    Not `hash(key)`: CPython randomizes string hashing per process unless
    `PYTHONHASHSEED` is pinned, so a corpus built twice would draw different video
    posteriors while the code claimed to be deterministic. blake2b has no such
    dependence on interpreter state.
    """
    import hashlib

    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big") % (2 ** 31)


# ----------------------------------------------------------------------------
# Decoding (CPU, in worker processes)
# ----------------------------------------------------------------------------
def decode_clip(payload: tuple[str, bytes], num_frames: int, size: int,
                sample_rate: int) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Decode one mp4 into `(num_frames, size, size, 3)` uint8 and stereo float32 audio.

    Runs in a worker process: PyAV holds non-picklable state, and decoding is the
    CPU-bound half of preprocessing while the VAEs are the GPU-bound half.

    Frames are taken from the middle of the clip so that the 10 s VGGSound window is
    not dominated by the fade-ins many YouTube clips start with. Video is resampled
    onto MiniMax-H3's own 24 fps by nearest-frame selection, the same "drop and
    duplicate whole frames" rule the reference uses for references.
    """
    name, blob = payload
    try:
        import av

        container = av.open(io.BytesIO(blob))
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if video_stream is None or audio_stream is None:
            return None

        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        if len(frames) < 8:
            return None
        source_fps = float(video_stream.average_rate or 25.0)

        # Centre window of `num_frames / 24` seconds, expressed in source frames.
        duration = num_frames / H.FPS
        want = int(round(duration * source_fps))
        start = max(0, (len(frames) - want) // 2)
        window = frames[start:start + want]
        if len(window) < 2:
            return None
        # Nearest-frame resample onto 24 fps.
        picks = np.clip(np.round(np.linspace(0, len(window) - 1, num_frames)).astype(int), 0, len(window) - 1)
        video = np.stack([window[i] for i in picks])            # (F, h, w, 3)

        # Short edge to `size`, then centre crop -- preserves aspect, no stretching.
        height, width = video.shape[1], video.shape[2]
        scale = size / min(height, width)
        new_h, new_w = max(size, int(round(height * scale))), max(size, int(round(width * scale)))
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear",
                                                 align_corners=False, antialias=True)
        top, left = (new_h - size) // 2, (new_w - size) // 2
        tensor = tensor[:, :, top:top + size, left:left + size]
        video = tensor.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

        # Audio: same centre window, resampled to the audio VAE's rate, forced stereo.
        container.seek(0)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
        container.close()
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=1)                   # (2, samples)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        audio = audio[:2]

        want_samples = int(round(duration * sample_rate))
        a_start = int(round(start / source_fps * sample_rate))
        segment = audio[:, a_start:a_start + want_samples]
        if segment.shape[1] < want_samples:
            segment = np.pad(segment, ((0, 0), (0, want_samples - segment.shape[1])))
        return name, video, segment.astype(np.float32)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Phase 1: text bank
# ----------------------------------------------------------------------------
@torch.no_grad()
def build_text_bank(ckpt: Path, labels: list[str], out_path: Path, max_tokens: int, device: str) -> None:
    """Encode one Context-IR prompt per class with the frozen Qwen3-VL conditioner.

    The prompt is tokenized **verbatim** -- no chat template, no special tokens --
    and the conditioning is `hidden_states[50]`, not the final one, because the last
    layer is post-norm and is not what the released weights were trained against.

    Every prompt is padded or truncated to `max_tokens`. MiniMax-H3 at inference
    packs one variable-length request with no padding and no attention mask; a
    training batch has to share one layout, so a fixed text length is the price of
    batching. Padding rows are real embeddings of the pad token, the same compromise
    a fixed 77-token CLIP context makes.
    """
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    print(f"[text] loading conditioner from {ckpt}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    processor = AutoProcessor.from_pretrained(ckpt / "processor")
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        ckpt / "text_encoder", dtype=torch.bfloat16
    ).to(device).eval()

    # Qwen3-VL's `mm_token_type_ids` tag every token as 0 text / 1 image / 2 video and
    # drive its per-modality rotary layout. The reference derives them with
    # `processor.create_mm_token_type_ids`, which only exists in transformers >= 5.
    # These prompts carry no vision block at all, so the tags are all zero by
    # construction and the processor call is not needed -- but the assertion below
    # has to stay, or a prompt that ever does carry an image would be silently
    # encoded as if it were plain text.
    vision_ids = {
        getattr(processor, name, None)
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
    } | {
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
        if token in tokenizer.get_vocab()
    }
    vision_ids.discard(None)

    def mm_token_type_ids_for(token_ids: list[int]) -> torch.Tensor:
        if vision_ids & set(token_ids):
            raise SystemExit(
                "[text] a prompt carries vision tokens; the all-text shortcut for "
                "`mm_token_type_ids` no longer holds. Use "
                "`processor.create_mm_token_type_ids` (transformers >= 5)."
            )
        return torch.zeros((1, len(token_ids)), dtype=torch.long, device=device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # One extra row holds the *unconditional* embedding. The released H3 is
    # guidance-distilled and has no negative prompt at all, so the null branch is not
    # part of the shipped model; it is needed here because H3-nano is trained with
    # classifier-free guidance first and distilled into a single-pass model second,
    # which is the pipeline that produces a checkpoint like the released one.
    prompts = [context_ir_prompt(label) for label in labels] + [""]
    embeds = torch.zeros(len(prompts), max_tokens, 5120, dtype=torch.float16)
    lengths = []
    t0 = time.time()
    for index, prompt in enumerate(prompts):
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        lengths.append(len(token_ids))
        token_ids = (token_ids[:max_tokens] + [pad_id] * max(0, max_tokens - len(token_ids)))[:max_tokens]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_types = mm_token_type_ids_for(token_ids)
        out = encoder.model(input_ids=input_ids, mm_token_type_ids=mm_types, output_hidden_states=True)
        embeds[index] = out.hidden_states[H.TEXT_ENCODER_LAYER][0].to(torch.float16).cpu()
        if index % 50 == 0:
            print(f"[text] {index}/{len(prompts)}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"labels": labels, "embeds": embeds, "max_tokens": max_tokens,
                "token_lengths": lengths, "template": CONTEXT_IR_TEMPLATE,
                "null_index": len(labels)}, out_path)
    print(f"[text] wrote {out_path}  {tuple(embeds.shape)}  (row {len(labels)} is the null prompt)  "
          f"prompt tokens min/median/max = {min(lengths)}/{int(np.median(lengths))}/{max(lengths)}", flush=True)


# ----------------------------------------------------------------------------
# Phase 2: media latents
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_media(ckpt: Path, tarballs: list[Path], label_of: dict[str, str], labels: list[str],
                 out_dir: Path, num_frames: int, size: int, per_shard: int,
                 decode_workers: int, device: str, limit: int | None) -> None:
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    print(f"[media] loading VAEs from {ckpt}", flush=True)
    # float32, not bfloat16. The reference's `encode_vae_condition` feeds
    # `pixels.to(torch.float32)` and this VAE keeps some biases in fp32 (the same
    # mixed-precision design as the transformer's `_keep_in_fp32_modules`), so a
    # bf16 input meets an fp32 bias:
    #   RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
    # At 20.8 GB in fp32 against 94 GB free, there is nothing to buy by narrowing it.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(ckpt / "vae", dtype=torch.float32).to(device).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(ckpt / "audio_vae", dtype=torch.float32).to(device).eval()

    pixel_mean = torch.tensor(H.PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(H.PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    a_mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    a_std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)

    label_index = {label: i for i, label in enumerate(labels)}
    expected_latent_frames = H.num_latent_frames_for(num_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_video, shard_audio, shard_label, shard_ids = [], [], [], []
    shard_index, kept, seen, skipped = 0, 0, 0, 0
    geometry_checked = False
    t0 = time.time()

    def flush() -> None:
        nonlocal shard_index, shard_video, shard_audio, shard_label, shard_ids
        if not shard_video:
            return
        path = out_dir / f"latents_{shard_index:04d}.pt"
        torch.save({
            "video": torch.stack(shard_video), "audio": torch.stack(shard_audio),
            "label": torch.tensor(shard_label, dtype=torch.int16), "ids": shard_ids,
            "num_frames": num_frames, "size": size,
        }, path)
        print(f"[media] wrote {path}  n={len(shard_video)}  "
              f"({time.time()-t0:.0f}s, {kept} kept / {seen} seen)", flush=True)
        shard_index += 1
        shard_video, shard_audio, shard_label, shard_ids = [], [], [], []

    # `spawn`, not the Linux default `fork`. The VAEs are already on CUDA by this
    # point, so the parent holds an initialised CUDA context, and CUDA is not
    # fork-safe: the children inherit a context they cannot use and hang the first
    # time anything touches the runtime -- which `decode_clip` does, via
    # `torch.nn.functional.interpolate`. The symptom is silent. The step sat at 0.3 %
    # CPU with the VAEs resident and the log frozen for thirteen minutes, which looks
    # exactly like a slow tarball rather than a deadlock.
    #
    # Spawned workers pay a one-off torch import each; against 12k clips that is
    # noise, and the single-process decode smoke test passing 6/6 while the pooled
    # version hung is the reason to be explicit about the start method here.
    import multiprocessing

    pool = ProcessPoolExecutor(max_workers=decode_workers,
                               mp_context=multiprocessing.get_context("spawn"))

    def drain(batch: list[tuple[str, bytes]]) -> bool:
        """Decode one batch in the worker pool and encode it on the GPU.

        Returns True when the requested clip limit has been reached. Factored out of
        the streaming loop so the *partial* batch at the end of a tarball is
        processed too; inlined, the tail was silently dropped, which is the kind of
        loss that never shows up as an error and only ever appears as a slightly
        smaller corpus than the log claims to have seen.
        """
        nonlocal kept, skipped, geometry_checked
        for result in pool.map(_decode_partial(num_frames, size), batch, chunksize=1):
            if result is None:
                skipped += 1
                continue
            name, video_np, audio_np = result
            ytid = name.split("|", 1)[1]

            pixels = torch.from_numpy(video_np).to(device).permute(3, 0, 1, 2)[None]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            posterior = vae.encode(pixels, return_dict=False)[0]
            # The released recipe samples the video posterior and rounds it through
            # float16; the seed is fixed at 42 for conditioning, and a per-clip seed
            # here keeps the corpus deterministic without giving every clip the same
            # posterior draw.
            latents = posterior.sample(
                generator=torch.Generator(device=device).manual_seed(stable_seed(ytid))
            ).float().cpu()
            latents = ((latents.to(torch.float16).float() - lat_mean) / lat_std)[0]

            if not geometry_checked:
                got = latents.shape[1]
                if got != expected_latent_frames:
                    raise SystemExit(
                        f"[media] FATAL latent-frame geometry mismatch: predicted "
                        f"{expected_latent_frames} for {num_frames} pixel frames "
                        f"(17n+5 -> 5n+2), the VAE produced {got}. Fix "
                        f"h3nano.num_latent_frames_for before training."
                    )
                print(f"[media] geometry OK: {num_frames} frames -> {got} latent frames, "
                      f"latent {tuple(latents.shape)}", flush=True)
                geometry_checked = True

            wave = torch.from_numpy(audio_np).to(device)[:, None]      # (2, 1, samples)
            a_post = audio_vae.encode(wave, return_dict=False)[0]
            # Soundtracks take the posterior mode and are never sampled.
            a_lat = a_post.mode().float().cpu().transpose(1, 2)        # (2, L, 32)
            a_lat = ((a_lat - a_mean) / a_std).transpose(1, 2)         # (2, 32, L)

            shard_video.append(latents.to(torch.float16))
            shard_audio.append(a_lat.to(torch.float16))
            shard_label.append(label_index[label_of[ytid]])
            shard_ids.append(ytid)
            kept += 1
            # Progress is logged on its own cadence, not on the shard cadence. Tying
            # the two together means the only way to see progress sooner is to write
            # smaller shards, i.e. paying Lustre metadata for observability -- the
            # same mistake as tying wandb logging to the checkpoint interval. With
            # per_shard=2000 this run was silent for the first half hour, and a
            # deadlock would have looked identical.
            if kept % 200 == 0:
                rate = kept / max(time.time() - t0, 1e-9)
                remaining = (limit - kept) / rate / 60 if limit else 0
                print(f"[media] {kept} kept / {seen} seen  {rate:.2f} clip/s"
                      + (f"  eta {remaining:.0f} min" if limit else ""), flush=True)
            if len(shard_video) >= per_shard:
                flush()
            if limit is not None and kept >= limit:
                flush()
                print(f"[media] limit {limit} reached", flush=True)
                return True
        return False

    try:
        for tarball in tarballs:
            print(f"[media] streaming {tarball}", flush=True)
            with tarfile.open(tarball, mode="r|gz") as tar:
                batch: list[tuple[str, bytes]] = []
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".mp4"):
                        continue
                    stem = Path(member.name).stem                       # "<ytid>_<start>"
                    ytid = stem.rsplit("_", 1)[0]
                    if ytid not in label_of:
                        continue
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    batch.append((stem + "|" + ytid, handle.read()))
                    seen += 1
                    if len(batch) < decode_workers * 2:
                        continue
                    if drain(batch):
                        return
                    batch = []
                if batch and drain(batch):
                    return
    finally:
        pool.shutdown(wait=True)
        flush()
        # Only claim a corpus that exists. This runs in a `finally`, so a crashed run
        # used to leave a manifest with kept=0 behind -- and the next run's
        # "corpus already present" guard believed it and skipped straight past an
        # empty dataset to training.
        if kept == 0:
            print("[media] kept 0 clips; not writing a manifest", flush=True)
            return
        manifest = out_dir / "manifest.json"
        manifest.write_text(json.dumps({
            "shards": shard_index, "kept": kept, "seen": seen, "skipped": skipped,
            "num_frames": num_frames, "size": size,
            "latent_frames": expected_latent_frames,
            "audio_latents": H.num_audio_latents_for(num_frames),
            "labels": labels,
        }, indent=2))
        print(f"[media] manifest: {kept} clips in {shard_index} shards "
              f"({skipped} undecodable of {seen})", flush=True)


class _decode_partial:
    """Picklable partial for the decode workers."""

    def __init__(self, num_frames: int, size: int):
        self.num_frames, self.size = num_frames, size

    def __call__(self, payload):
        return decode_clip(payload, self.num_frames, self.size, H.AUDIO_SAMPLING_RATE)


# ----------------------------------------------------------------------------
def verify_environment(require_cuda: bool = True) -> None:
    """Fail in the first seconds of a job rather than three hours into it.

    This lives in Python, not in the batch script, on purpose: `sbatch` snapshots
    the batch script at submission, so a fix to it never reaches a job that is
    already queued, while an external module is read at run time and does. Every
    import below is one the pipeline needs later, and each has already broken once.
    """
    problems = []
    try:
        import diffusers
        from diffusers import (  # noqa: F401
            AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
            MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel,
        )
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: F401
            MiniMaxH3PrepareLayoutStep,
        )
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # noqa: F401
            align_num_frames,
        )
        # These two are the ones job 5998320 actually died on, and neither was covered
        # by the class imports above: `ModularPipeline.from_pretrained` pulls in
        # `DiffusionPipeline`, which pulls `diffusers.pipelines.pipeline_utils`, which
        # needs `get_cached_repo_tree` from huggingface_hub >= 1.x. Checking only the
        # imports one happens to think of is how a version mismatch reaches a compute
        # node; check the path the job takes.
        from diffusers import DiffusionPipeline, ModularPipeline  # noqa: F401
        import diffusers.pipelines.pipeline_utils  # noqa: F401
        print(f"[env] diffusers {diffusers.__version__} with MiniMax-H3 and pipeline_utils OK",
              flush=True)
    except Exception as exc:
        problems.append(f"diffusers: {type(exc).__name__}: {exc}")
    try:
        # peft is on the import path to the H3 transformer via `PeftAdapterMixin`, and
        # peft >= 0.18 probes transformer_engine, whose import needs libnvrtc from
        # LD_LIBRARY_PATH. Importing it here turns that into a first-minute failure
        # with a clear name instead of a RuntimeError three frames deep in diffusers.
        import peft
        print(f"[env] peft {peft.__version__} OK", flush=True)
    except Exception as exc:
        problems.append(f"peft: {type(exc).__name__}: {exc} "
                        f"(is LD_LIBRARY_PATH carrying the nvidia lib dirs? see code/env.sh)")
    try:
        import transformers
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
        print(f"[env] transformers {transformers.__version__} with Qwen3-VL OK", flush=True)
    except Exception as exc:
        problems.append(f"transformers: {type(exc).__name__}: {exc}")
    try:
        import av
        print(f"[env] PyAV {av.__version__} OK", flush=True)
    except Exception as exc:
        problems.append(f"PyAV: {type(exc).__name__}: {exc}")
    # CUDA is required only when the caller asked for it. Running the whole media
    # path on CPU is slow but valid, and it is the only way to exercise the real VAE
    # weights -- including the 17n+5 -> 5n+2 geometry assertion -- when every card in
    # reach is held by someone else.
    if require_cuda and not torch.cuda.is_available():
        problems.append("no CUDA device visible")
    if problems:
        raise SystemExit("[env] FATAL:\n  " + "\n  ".join(problems))

    # Then the conventions themselves. These are seconds of CPU arithmetic and they
    # catch the class of bug that costs the most: a flipped sign in the flow-matching
    # convention, a transposed audio axis, a timestep the architecture cannot express.
    # None of those raise on their own and none show up in a loss curve, so finding
    # them at minute one instead of hour three is worth the seconds.
    try:
        import test_h3nano

        failures = test_h3nano.main()
    except Exception as exc:
        raise SystemExit(f"[env] FATAL: convention checks could not run: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("[env] FATAL: convention checks failed; training on this would be worthless")
    print("[env] convention checks passed", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=["text", "media", "both"], default="both")
    parser.add_argument("--shards", default="00,01")
    parser.add_argument("--num-frames", type=int, default=73, help="Snapped to 17n+5")
    parser.add_argument("--size", type=int, default=256, help="Square canvas, a multiple of 32")
    parser.add_argument("--max-text-tokens", type=int, default=96)
    parser.add_argument("--per-shard", type=int, default=2000)
    parser.add_argument("--decode-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    verify_environment(require_cuda=args.device.startswith('cuda'))

    root = Path(args.root)
    ckpt = root / "ckpt" / "h3"
    data = root / "data" / "vggsound"
    out = root / "data" / "latents"
    out.mkdir(parents=True, exist_ok=True)

    num_frames = H.snap_num_frames(args.num_frames)
    if num_frames != args.num_frames:
        print(f"[main] num_frames {args.num_frames} -> {num_frames} (17n+5)", flush=True)
    if args.size % 32:
        raise SystemExit(f"--size must be a multiple of 32 (16x VAE * 2 patch), got {args.size}")

    with open(data / "vggsound.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    label_of = {row[0]: row[2] for row in rows if len(row) >= 3}
    labels = sorted({row[2] for row in rows if len(row) >= 3})
    print(f"[main] {len(label_of)} clips over {len(labels)} classes", flush=True)

    if args.phase in ("text", "both"):
        build_text_bank(ckpt, labels, out / "text_bank.pt", args.max_text_tokens,
                        pick_device(64.0, args.device))
        # The conditioner is 62 GB and is never needed again; release it before the VAEs load.
        torch.cuda.empty_cache()

    if args.phase in ("media", "both"):
        tarballs = [data / f"vggsound_{int(s):02d}.tar.gz" for s in args.shards.split(",") if s.strip()]
        tarballs = [t for t in tarballs if t.exists()]
        if not tarballs:
            raise SystemExit(f"[main] no shard tarballs found under {data}")
        encode_media(ckpt, tarballs, label_of, labels, out, num_frames, args.size,
                     args.per_shard, args.decode_workers, pick_device(16.0, args.device),
                     args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
