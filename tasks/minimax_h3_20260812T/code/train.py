#!/usr/bin/env python
"""Train H3-nano end to end: pretraining, task post-training, and guidance distillation.

MiniMax released H3's weights and nothing about how they were made, so the three
stages here are reconstructed from what the released artefacts *state* rather than
from a recipe:

``pretrain``
    Text-to-video-and-audio from random init. Rectified flow in H3's own
    convention, two independent shifted timestep schedules (``shift = 12`` video,
    ``shift = 3`` audio), and classifier-free guidance dropout so a guided sampler
    exists to distill from later.

``sft_fl2va``
    The released FL2VA checkpoint conditions on a first and/or last keyframe. The
    anchors are packed as leading video rows held at ``t = 0.999`` -- the diffusers
    port documents that "the released model was trained with its anchors very
    slightly noised, so conditioning on exactly t = 1.0 is off-distribution", which
    is a statement about MiniMax's training, and this stage reproduces it.

``distill_cfg``
    The released checkpoints are guidance-distilled: "guidance is baked into the
    weights, so there is no guider, no negative_prompt and no guidance_scale, and
    every step runs exactly one forward pass." This stage turns the two-pass guided
    model of the previous stages into exactly such a one-pass model, and the
    measured halving of sampling cost is the check that it worked.

Operationally the script obeys this cluster's rules: checkpoints are written to
node-local ``$TMPDIR`` and synced to Lustre at the end, resume goes through an
atomically-written ``latest_checkpoint.json`` breadcrumb rather than any listing
of the checkpoint directory, and W&B runs online with its own directory on the
node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3nano as H  # noqa: E402


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class LatentCorpus:
    """Every latent shard, resident in host RAM.

    The whole corpus is a few GB against the node's ~856 GB, and a GH200's
    NVLink-C2C link to that memory runs at ~450 GB/s, so holding it host-side and
    slicing per step costs nothing measurable. It also means the training loop
    never touches Lustre after startup, which is the point.
    """

    def __init__(self, latent_dir: Path, rank: int = 0):
        manifest = json.loads((latent_dir / "manifest.json").read_text())
        self.manifest = manifest
        shards = [latent_dir / f"latents_{i:04d}.pt" for i in range(manifest["shards"])]
        videos, audios, labels = [], [], []
        for shard in shards:
            if not shard.exists():
                continue
            blob = torch.load(shard, map_location="cpu", weights_only=False)
            videos.append(blob["video"])
            audios.append(blob["audio"])
            labels.append(blob["label"])
        if not videos:
            raise SystemExit(f"no latent shards under {latent_dir}")
        self.video = torch.cat(videos)                      # (N, 24, F, h, w) fp16
        self.audio = torch.cat(audios)                      # (N, 2, 32, L)    fp16
        self.label = torch.cat(labels).long()               # (N,)
        bank = torch.load(latent_dir / "text_bank.pt", map_location="cpu", weights_only=False)
        self.text = bank["embeds"]                          # (C+1, T, 5120) fp16
        self.null_index = bank["null_index"]
        self.labels = bank["labels"]
        if rank == 0:
            print(f"[data] {len(self.video)} clips | video {tuple(self.video.shape)} "
                  f"| audio {tuple(self.audio.shape)} | text {tuple(self.text.shape)} "
                  f"| {len(self.labels)} classes", flush=True)

    def __len__(self) -> int:
        return len(self.video)

    def sample(self, batch: int, device, generator, cfg_dropout: float = 0.0):
        index = torch.randint(len(self.video), (batch,), generator=generator)
        labels = self.label[index].clone()
        if cfg_dropout > 0:
            drop = torch.rand(batch, generator=generator) < cfg_dropout
            labels[drop] = self.null_index
        return (
            self.video[index].to(device, torch.float32, non_blocking=True),
            self.audio[index].to(device, torch.float32, non_blocking=True),
            self.text[labels].to(device, torch.float32, non_blocking=True),
            labels,
        )


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------
def save_checkpoint(ckpt_dir: Path, step: int, model, optimizer, scheduler_state: dict,
                    config: dict, keep: int = 3) -> None:
    """Write a checkpoint and then, atomically, the breadcrumb that names it.

    The breadcrumb exists so that resume never has to *list* the checkpoint
    directory. `ls -1td ckpt/` stats every entry on every call, which is the
    metadata pattern that got this group's jobs suspended once; a single
    `latest_checkpoint.json`, written to a temp file and renamed, costs one open.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
    torch.save({"step": step, "model": state, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler_state, "config": config}, path)

    tmp = ckpt_dir / ".latest_checkpoint.json.tmp"
    tmp.write_text(json.dumps({"step": step, "path": str(path),
                               "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    os.replace(tmp, ckpt_dir / "latest_checkpoint.json")

    # Prune by the step numbers we ourselves wrote, tracked in the breadcrumb's
    # sibling ledger, so pruning also never lists the directory.
    ledger = ckpt_dir / "written_steps.json"
    steps = json.loads(ledger.read_text()) if ledger.exists() else []
    steps.append(step)
    for old in steps[:-keep]:
        stale = ckpt_dir / f"step_{old:08d}.pt"
        if stale.exists() and old != step:
            stale.unlink()
    ledger.write_text(json.dumps(steps[-keep:]))


def load_checkpoint(ckpt_dir: Path, model, optimizer=None, device="cpu"):
    breadcrumb = ckpt_dir / "latest_checkpoint.json"
    if not breadcrumb.exists():
        return 0
    info = json.loads(breadcrumb.read_text())
    blob = torch.load(info["path"], map_location=device, weights_only=False)
    target = model.module if isinstance(model, DistributedDataParallel) else model
    target.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    print(f"[ckpt] resumed from {info['path']} at step {blob['step']}", flush=True)
    return int(blob["step"])


# ----------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------
def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def make_condition_rows(video_latents: torch.Tensor, anchors: tuple[str, ...],
                        generator=None, anchor_t: float = H.KEYFRAME_NOISE_AUG) -> torch.Tensor:
    """Build the keyframe anchor rows of an `fl2va` batch.

    The anchors are the clean latent frames themselves, noised to `t = 0.999` --
    what `keyframe_noise_aug` prescribes -- and then patchified. They lead the video
    rows in the packed sequence and never contribute to the loss.

    `anchor_t` is a parameter rather than the constant so that evaluation can sweep
    it: if 0.999 really is a property of how the released model was trained, a model
    trained the same way should degrade at exactly `t = 1.0`, which it never saw.
    """
    rows = []
    for anchor in anchors:
        frame = video_latents[:, :, :1] if anchor == "first" else video_latents[:, :, -1:]
        noise = torch.randn(frame.shape, device=frame.device, dtype=frame.dtype, generator=generator)
        noised = H.add_noise(frame, noise, torch.full((frame.shape[0],), anchor_t, device=frame.device))
        rows.append(H.patchify_video_latents(noised))
    return torch.cat(rows, dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=["pretrain", "sft_fl2va", "distill_cfg"], default="pretrain")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", choices=["micro", "nano", "small"], default="nano")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=8, help="Per GPU")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--audio-weight", type=float, default=1.0)
    parser.add_argument("--timestep-mode", choices=["uniform", "logit_normal"], default="uniform")
    parser.add_argument("--video-shift", type=float, default=H.VIDEO_FLOW_SHIFT,
                        help="Video sigma shift; 12.0 is the released checkpoint's")
    parser.add_argument("--audio-shift", type=float, default=H.AUDIO_FLOW_SHIFT,
                        help="Audio sigma shift; 3.0 is the released checkpoint's")
    parser.add_argument("--anchors", default="first", help="fl2va anchors, e.g. 'first' or 'first,last'")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="w baked in by distill_cfg")
    parser.add_argument("--teacher", default=None, help="Checkpoint dir the stage initializes/distills from")
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--wandb", default="h3-nano")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    run_name = args.run_name or f"{args.stage}-{args.model}-{os.environ.get('SLURM_JOB_ID', 'local')}"

    # -------------------------------------------------------------- distributed
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # CPU is supported only so the whole loop can be dry-run before an allocation is
    # spent on it; nothing about the recipe changes with the device.
    has_cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if has_cuda else "gloo")
    if has_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if has_cuda else "cpu")
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    generator = torch.Generator().manual_seed(args.seed + rank)

    # -------------------------------------------------------------- data, model
    corpus = LatentCorpus(root / "data" / "latents", rank=rank)
    config = {"micro": H.MICRO_CONFIG, "nano": H.NANO_CONFIG, "small": H.SMALL_CONFIG}[args.model]

    model = H.build_transformer(config).to(device)
    model.enable_gradient_checkpointing()
    if is_main:
        census = H.parameter_census(model)
        print(f"[model] H3-nano/{args.model}: {census['TOTAL']/1e6:.2f} M parameters", flush=True)
        for key, value in sorted(census.items(), key=lambda kv: -kv[1]):
            if key != "TOTAL":
                print(f"[model]   {key:20s} {value/1e6:8.3f} M ({100*value/census['TOTAL']:4.1f}%)", flush=True)

    teacher = None
    if args.stage in ("sft_fl2va", "distill_cfg"):
        if not args.teacher:
            raise SystemExit(f"--stage {args.stage} needs --teacher <checkpoint dir>")
        load_checkpoint(Path(args.teacher), model, None, device)
        if args.stage == "distill_cfg":
            # The teacher is the guided sampler being collapsed into one pass; it is
            # the same weights, frozen, evaluated twice per step.
            teacher = H.build_transformer(config).to(device)
            load_checkpoint(Path(args.teacher), teacher, None, device)
            teacher.eval().requires_grad_(False)

    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay, eps=1e-8)

    # -------------------------------------------------------------- geometry
    manifest = corpus.manifest
    num_frames, size = manifest["num_frames"], manifest["size"]
    latent_frames = corpus.video.shape[2]
    latent_h, latent_w = corpus.video.shape[3], corpus.video.shape[4]
    audio_latents = corpus.audio.shape[3]
    text_tokens = corpus.text.shape[1]
    anchors = tuple(a for a in args.anchors.split(",") if a) if args.stage == "sft_fl2va" else ()

    layout = H.build_layout(text_tokens, latent_frames, latent_h, latent_w,
                            audio_latents, keyframe_anchors=anchors).to(device)
    if is_main:
        print(f"[layout] {num_frames} frames @ {size}px -> {latent_frames}x{latent_h}x{latent_w} latents, "
              f"{audio_latents} audio latents/channel", flush=True)
        print(f"[layout] sequence_length={layout.sequence_length} "
              f"(video {len(layout.video_indices)}, text {len(layout.text_indices)}, "
              f"audio {len(layout.audio_indices)}; {layout.num_condition_video_rows} conditioning)", flush=True)

    # -------------------------------------------------------------- checkpoint dirs
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"h3nano_{run_name}"
    ckpt_dir = tmp_root / "checkpoints"
    final_dir = root / "runs" / run_name
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        # Resume prefers a checkpoint already synced to Lustre from a previous
        # allocation; `$TMPDIR` does not survive the job that created it.
        for candidate in (ckpt_dir, final_dir / "checkpoints"):
            if (candidate / "latest_checkpoint.json").exists():
                ckpt_dir = candidate
                break
    start_step = load_checkpoint(ckpt_dir, model, optimizer, device) if is_main else 0
    if world > 1:
        marker = torch.tensor([start_step], device=device)
        dist.broadcast(marker, 0)
        start_step = int(marker.item())
        if start_step > 0 and not is_main:
            load_checkpoint(ckpt_dir, model, optimizer, device)

    # -------------------------------------------------------------- wandb
    wandb_run = None
    if is_main and not args.no_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_DIR", str(tmp_root))
            os.environ["WANDB_MODE"] = "online"
            wandb_run = wandb.init(project=args.wandb, name=run_name, entity="oxford-lob",
                                   config={**vars(args), **{f"arch_{k}": v for k, v in config.items()},
                                           "sequence_length": layout.sequence_length,
                                           "clips": len(corpus)})
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as exc:
            print(f"[wandb] disabled ({exc})", flush=True)

    # -------------------------------------------------------------- loop
    if is_main:
        print(f"[train] stage={args.stage} steps={start_step}->{args.steps} "
              f"batch={args.batch_size}/gpu x {world} gpu", flush=True)
    model.train()
    window, t_start = [], time.time()

    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.lr, args.warmup, args.steps)

        # `distill_cfg` needs the *conditional* text for the student and both
        # branches for the teacher, so it draws without dropout and builds the null
        # branch explicitly.
        dropout = 0.0 if args.stage == "distill_cfg" else args.cfg_dropout
        video, audio, text, labels = corpus.sample(args.batch_size, device, generator, dropout)

        condition_rows = make_condition_rows(video, anchors) if anchors else None
        batch = H.make_flow_batch(video, audio, text, layout, timestep_mode=args.timestep_mode,
                                  condition_rows=condition_rows,
                                  video_shift=args.video_shift, audio_shift=args.audio_shift)

        if args.stage == "distill_cfg":
            null_text = corpus.text[torch.full_like(labels, corpus.null_index)].to(device, torch.float32)
            with torch.no_grad():
                cond_v, cond_a = teacher(**batch.transformer_kwargs())
                uncond_kwargs = dict(batch.transformer_kwargs())
                uncond_kwargs["encoder_hidden_states"] = null_text
                unc_v, unc_a = teacher(**uncond_kwargs)
                w = args.guidance_scale
                batch.video_target = unc_v + w * (cond_v - unc_v)
                batch.audio_target = unc_a + w * (cond_a - unc_a)

        video_pred, audio_pred = model(**batch.transformer_kwargs())
        loss, logs = H.flow_loss(video_pred, audio_pred, batch, audio_weight=args.audio_weight)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        window.append(logs)
        if is_main and (step + 1) % args.log_every == 0:
            mean = {k: sum(d[k] for d in window) / len(window) for k in window[0]}
            elapsed = time.time() - t_start
            rate = (step + 1 - start_step) / elapsed
            remaining = (args.steps - step - 1) / max(rate, 1e-9) / 3600
            print(f"[{step+1:>7}/{args.steps}] loss {mean['loss']:.4f} "
                  f"(v {mean['loss_video']:.4f} a {mean['loss_audio']:.4f}) "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} gn {float(grad_norm):.2f} "
                  f"{rate:.2f} it/s  eta {remaining:.2f}h", flush=True)
            if wandb_run is not None:
                wandb_run.log({**mean, "lr": optimizer.param_groups[0]["lr"],
                               "grad_norm": float(grad_norm), "it_per_s": rate}, step=step + 1)
            window = []

        if is_main and (step + 1) % args.ckpt_every == 0:
            save_checkpoint(ckpt_dir, step + 1, model, optimizer, {}, {**vars(args), "arch": config})

    # -------------------------------------------------------------- finish
    if is_main:
        save_checkpoint(ckpt_dir, args.steps, model, optimizer, {}, {**vars(args), "arch": config})
        if ckpt_dir != final_dir / "checkpoints":
            print(f"[sync] {ckpt_dir} -> {final_dir}/checkpoints", flush=True)
            subprocess.run(["rsync", "-a", f"{ckpt_dir}/", str(final_dir / "checkpoints") + "/"], check=False)
        (final_dir / "train_config.json").write_text(json.dumps(
            {**vars(args), "arch": config, "sequence_length": layout.sequence_length,
             "clips": len(corpus), "world_size": world}, indent=2))
        print(f"[train] done -> {final_dir}", flush=True)
        if wandb_run is not None:
            wandb_run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
