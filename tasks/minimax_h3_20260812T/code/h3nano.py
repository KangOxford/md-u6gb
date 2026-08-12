#!/usr/bin/env python
"""H3-nano: a scaled-down MiniMax-H3 that is architecturally the released model.

MiniMax released H3's weights but neither a technical report nor any training
code, so this module is split along that line:

* **The architecture is not reimplemented.** `MiniMaxH3Transformer3DModel` from
  diffusers *is* the reference model, and every dimension of it is a constructor
  argument, so H3-nano is that class under a smaller config. Reimplementing it
  by hand would only add a way to be subtly wrong. The same goes for the packed
  layout: `MiniMaxH3PrepareLayoutStep.build_packed_sequence` is a pure static
  method, so training builds its sequences with the very function inference uses.

* **The training recipe is reconstructed here**, because it was never released.
  Every rule below is derived from something checkable in the released artefacts
  (the scheduler's algebra, the inference-time row/timestep assignment, the
  documented `keyframe_noise_aug`), and each is labelled with what it rests on.

The scaling keeps every *ratio* of the 33B model, so the small model differs
from H3 in size and in nothing else:

    quantity              H3 (33B)   H3-nano     ratio held
    hidden_size              5376        576     --
    attn inner / hidden   7168/5376   768/576    4/3   (attention is wider than
                                                        the residual stream)
    ffn_dim / hidden     14336/5376  1536/576    8/3
    time_embed_dim/hid    2688/5376   288/576    1/2
    rotated / head_dim      96/128      48/64    3/4   (rope_freq_dim 16 -> 8)
    num_layers                 50         12     depth is the one free knob
    in_channels / audio     24 / 32    24 / 32   identical: the real frozen VAEs
    text_dim                 5120       5120     identical: the real Qwen3-VL-32B

Holding `in_channels`, `audio_in_channels` and `text_dim` fixed is what makes the
comparison meaningful: H3-nano denoises latents in *the same latent space* as
the released model, conditioned on *the same text features*, so any difference
in behaviour is the transformer's, not the tokenizer's or the conditioner's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# ----------------------------------------------------------------------------
# Model facts of the released checkpoint. Read off `transformer/config.json`,
# `scheduler/scheduler_config.json` and the MiniMaxH3ModularPipeline properties.
# ----------------------------------------------------------------------------
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2      # MINIMAX_H3_MODALITY_NUM == 3
FPS = 24.0
AUDIO_SAMPLING_RATE = 32_000
AUDIO_LATENT_RATE = 40                        # audio latents per second, per channel
AUDIO_CHANNELS = 2                            # stereo, packed channel-major
VAE_SPATIAL_COMPRESSION = 16
VAE_FRAMES_PER_CHUNK = 17                     # 17 pixel frames -> 5 latent frames
VAE_LATENTS_PER_CHUNK = 5
VIDEO_FLOW_SHIFT = 12.0                       # scheduler/scheduler_config.json
AUDIO_FLOW_SHIFT = 3.0                        # audio_scheduler/scheduler_config.json
KEYFRAME_NOISE_AUG = 0.999                    # anchors are held just short of clean
TEXT_ENCODER_LAYER = 50                       # Qwen3-VL hidden_states[50], not the last
PIXEL_MEAN = (0.485, 0.456, 0.406)            # the video VAE normalizes with ImageNet's
PIXEL_STD = (0.229, 0.224, 0.225)

# H3-nano. Ratios as tabulated in the module docstring.
NANO_CONFIG = dict(
    num_attention_heads=12,
    attention_head_dim=64,
    hidden_size=576,
    num_layers=12,
    num_refiner_layers=2,
    ffn_dim=1536,
    in_channels=24,
    audio_in_channels=32,
    patch_size=(1, 2, 2),
    text_dim=5120,
    freq_dim=256,
    time_embed_hidden_dim=576,
    time_embed_dim=288,
    rope_freq_dim=8,
    rope_theta=10000.0,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)

# A second, wider point on the same ratio family, for the scaling comparison.
MICRO_CONFIG = dict(NANO_CONFIG, hidden_size=384, num_layers=8, num_attention_heads=8,
                    ffn_dim=1024, time_embed_hidden_dim=384, time_embed_dim=192)
SMALL_CONFIG = dict(NANO_CONFIG, hidden_size=768, num_layers=16, num_attention_heads=16,
                    ffn_dim=2048, time_embed_hidden_dim=768, time_embed_dim=384)


def build_transformer(config: dict | None = None, dtype: torch.dtype = torch.float32):
    """Instantiate H3-nano: the reference class, small config, freshly initialized."""
    from diffusers import MiniMaxH3Transformer3DModel

    model = MiniMaxH3Transformer3DModel(**(config or NANO_CONFIG))
    return model.to(dtype)


def parameter_census(model) -> dict[str, int]:
    """Split parameters the way MiniMax reports them, so the split is checkable."""
    buckets: dict[str, int] = {}
    for name, param in model.named_parameters():
        # `token_refiner` has to be tested before the generic `.attn.` / `.ff.` rules,
        # or its two blocks are silently counted as part of the main stack and the
        # refiner appears to hold only its final norm.
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
    buckets["TOTAL"] = sum(buckets.values())
    return buckets


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def snap_num_frames(num_frames: int) -> int:
    """Round up to the next frame count the video VAE can decode, i.e. `17n + 5`."""
    if num_frames <= 5:
        return 5
    return 17 * math.ceil((num_frames - 5) / 17) + 5


def num_latent_frames_for(num_frames: int) -> int:
    """Latent frames a `17n + 5` pixel-frame clip decodes from.

    The VAE groups 17 pixel frames into 5 latent frames, and the `+5` tail is one
    further group's worth of leading frames, so `17n + 5` pixel frames become
    `5n + 2` latents. `_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)` is the same
    grouping seen from the rotary clock: `1 + 4 + 4 + 4 + 4 == 17`.

    This is the *predicted* relation; `preprocess_vggsound.py` asserts it against
    the real VAE on the first clip it encodes and fails loudly if it is wrong.
    """
    return 5 * ((num_frames - 5) // 17) + 2


def num_audio_latents_for(num_frames: int) -> int:
    """Audio latents per channel covering the same wall-clock as `num_frames`."""
    return int(round(num_frames / FPS * AUDIO_LATENT_RATE))


def patchify_video_latents(latents: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """`(B, C, F, H, W)` -> `(B, num_rows, C * prod(patch))`, frame-major then row-major.

    Same permutation as `diffusers...before_denoise.patchify_video_latents`, but the
    batch axis is kept: the reference flattens it because a request is one sequence,
    while training runs many replicas of one layout.
    """
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    latents = latents.reshape(batch, channels, frames // patch_t, patch_t,
                              height // patch_h, patch_h, width // patch_w, patch_w)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(batch, -1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_latents(rows: torch.Tensor, frames: int, latent_h: int, latent_w: int,
                             channels: int = 24, patch_size=(1, 2, 2)) -> torch.Tensor:
    """Inverse of `patchify_video_latents`."""
    patch_t, patch_h, patch_w = patch_size
    batch = rows.shape[0]
    rows = rows.reshape(batch, frames // patch_t, latent_h // patch_h, latent_w // patch_w,
                        channels, patch_t, patch_h, patch_w)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(batch, channels, frames, latent_h, latent_w).contiguous()


def pack_audio_channel_major(audio_latents: torch.Tensor) -> torch.Tensor:
    """`(B, channels, C_lat, L)` -> `(B, channels * L, C_lat)`, channel-major.

    Channel-major is what `_fill_audio_positions` assumes: it writes the rotary time
    as `arange(L).repeat(channels)` and pins the width coordinate to the first grid
    value for the first `L` rows and the last for the rest, i.e. left channel then
    right channel, each in time order.
    """
    batch, channels, latent_channels, length = audio_latents.shape
    return audio_latents.permute(0, 1, 3, 2).reshape(batch, channels * length, latent_channels).contiguous()


def unpack_audio_channel_major(rows: torch.Tensor, channels: int = AUDIO_CHANNELS) -> torch.Tensor:
    """Inverse of `pack_audio_channel_major`."""
    batch, total, latent_channels = rows.shape
    length = total // channels
    return rows.reshape(batch, channels, length, latent_channels).permute(0, 1, 3, 2).contiguous()


# ----------------------------------------------------------------------------
# Rectified flow, in MiniMax-H3's convention
# ----------------------------------------------------------------------------
def apply_shift(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """`sigma' = s*sigma / (1 + (s-1)*sigma)`, the scheduler's exponential shift."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def sample_timesteps(batch: int, shift: float, device, generator=None,
                     mode: str = "uniform") -> torch.Tensor:
    """Draw training timesteps in H3's convention, where `t = 1` is clean.

    `set_timesteps` builds its grid as `sigma = shift(linspace(1, 0, N))`, so drawing
    `u ~ U(0, 1)`, shifting it and setting `t = 1 - sigma` reproduces exactly the
    marginal the sampler will walk at inference. With no published training recipe
    that match is the defensible default, and it is what makes `shift = 12` for video
    and `shift = 3` for audio mean the same thing in training as in sampling: most of
    the probability mass sits at high sigma, i.e. the noisy end, and video gets far
    more of it than audio.

    `mode="logit_normal"` is the SD3 alternative, kept for the ablation rather than
    for the default.
    """
    if mode == "uniform":
        u = torch.rand(batch, device=device, generator=generator)
    elif mode == "logit_normal":
        u = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    else:
        raise ValueError(f"unknown timestep sampling mode {mode!r}")
    return 1.0 - apply_shift(u, shift)


def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`x_t = t*x0 + (1 - t)*noise` -- `MiniMaxH3Scheduler.scale_noise`, verbatim."""
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return t * x0 + (1.0 - t) * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """The velocity H3 predicts, which is data-ward and therefore `x0 - noise`.

    `MiniMaxH3Scheduler.step` denoises with `x0 = x_t + sigma * v`, the opposite sign
    to the usual flow-match convention. Substituting the forward process:

        v = (x0 - x_t) / (1 - t)
          = (x0 - t*x0 - (1 - t)*noise) / (1 - t)
          = x0 - noise

    which is independent of `t`, so nothing about the schedule enters the target.
    """
    return x0 - noise


# ----------------------------------------------------------------------------
# The packed sequence, at training time
# ----------------------------------------------------------------------------
@dataclass
class PackedLayout:
    """One packed layout, shared by every item of a batch.

    The transformer treats its batch axis as a pure replication axis: `token_tags`,
    `position_ids` and the row-index tensors describe *one* sequence. So a training
    batch has to be homogeneous in geometry -- same resolution, same duration, same
    text length -- which is why the text stream is padded to a fixed length rather
    than masked.
    """

    position_ids: torch.Tensor          # (S, 3) float64 -> cast on use
    token_tags: torch.Tensor            # (S,)
    video_indices: torch.Tensor         # (N_video,)
    audio_indices: torch.Tensor         # (N_audio,)
    text_indices: torch.Tensor          # (N_text,)
    num_condition_video_rows: int
    num_condition_audio_rows: int
    sequence_length: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    rows_per_frame: int
    keyframe_anchors: tuple[str, ...] = ()

    def to(self, device) -> "PackedLayout":
        return PackedLayout(
            position_ids=self.position_ids.to(device),
            token_tags=self.token_tags.to(device),
            video_indices=self.video_indices.to(device),
            audio_indices=self.audio_indices.to(device),
            text_indices=self.text_indices.to(device),
            num_condition_video_rows=self.num_condition_video_rows,
            num_condition_audio_rows=self.num_condition_audio_rows,
            sequence_length=self.sequence_length,
            num_latent_frames=self.num_latent_frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            num_audio_latents=self.num_audio_latents,
            rows_per_frame=self.rows_per_frame,
            keyframe_anchors=self.keyframe_anchors,
        )


def build_layout(num_text_tokens: int, num_latent_frames: int, latent_height: int,
                 latent_width: int, num_audio_latents: int, patch_size=(1, 2, 2),
                 keyframe_anchors: tuple[str, ...] = ()) -> PackedLayout:
    """Build the training layout with the *reference* packing function.

    `MiniMaxH3PrepareLayoutStep.build_packed_sequence` is a pure static method with no
    pipeline state, so calling it here means training and inference cannot drift: the
    aspect-normalized spatial grid, the non-uniform `5/3 * (1, 4, 4, 4, 4)` rotary
    clock and the channel-major audio pinning are the released implementation's, not a
    paraphrase of it.
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    # Every text row is tagged TEXT except a keyframe's vision block, which H3 tags
    # VIDEO. Pretraining conditions on text alone, so the tags are uniform here.
    text_token_tags = torch.full((num_text_tokens,), TEXT_TAG, dtype=torch.long)

    (position_ids, token_tags, video_indices, audio_indices, text_indices,
     num_condition_video_rows, num_condition_audio_rows) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        patch_size=patch_size,
        audio_channels=AUDIO_CHANNELS,
        audio_tag=AUDIO_TAG,
        video_tag=VIDEO_TAG,
        keyframe_anchors=keyframe_anchors,
    )
    patch_h, patch_w = patch_size[1], patch_size[2]
    return PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
        sequence_length=int(token_tags.numel()),
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        rows_per_frame=(latent_height // patch_h) * (latent_width // patch_w),
        keyframe_anchors=keyframe_anchors,
    )


def build_row_timesteps(layout: PackedLayout, video_t: float, audio_t: float,
                        condition_video_t: float | None = None,
                        condition_audio_t: float | None = None):
    """Assign every row its timestep, exactly as the sampler does.

    Transcribed from `MiniMaxH3PrepareRowTimestepsStep.build_row_timesteps`. Two
    consequences matter for training and are easy to get wrong:

    * **Text rows inherit the video timestep.** The reference fills the whole
      sequence with `video_timestep` first and only then overwrites the audio and
      conditioning rows, so a text row's AdaLN modulation moves with the video's
      noise level. Text conditioning is not static across the trajectory.
    * **Conditioning rows sit at `max(t, 0.999)`**, not at 1.0. The released model
      was trained with slightly noised anchors, so exact `t = 1` is off-distribution.

    Returns the `(unique_timesteps, timestep_indices)` pair the transformer takes;
    reducing to unique values is what keeps the timestep MLP cheap when a sequence
    carries three or four distinct noise levels.
    """
    row_timesteps = torch.full((layout.sequence_length,), float(video_t), dtype=torch.float32)
    n_cond_v, n_cond_a = layout.num_condition_video_rows, layout.num_condition_audio_rows
    if n_cond_v:
        row_timesteps[layout.video_indices[:n_cond_v]] = float(
            condition_video_t if condition_video_t is not None else max(video_t, KEYFRAME_NOISE_AUG)
        )
    row_timesteps[layout.audio_indices[n_cond_a:]] = float(audio_t)
    if n_cond_a:
        row_timesteps[layout.audio_indices[:n_cond_a]] = float(
            condition_audio_t if condition_audio_t is not None else max(audio_t, KEYFRAME_NOISE_AUG)
        )
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


@dataclass
class FlowBatch:
    """Everything one optimizer step needs, already in transformer argument order."""

    video_rows: torch.Tensor          # (B, N_video, 96)  noised, conditioning rows first
    audio_rows: torch.Tensor          # (B, N_audio, 32)  noised, reference rows first
    text_embeds: torch.Tensor         # (B, N_text, 5120)
    timestep: torch.Tensor            # (num_distinct,)
    timestep_indices: torch.Tensor    # (S,)
    video_target: torch.Tensor        # (B, N_video, 96)
    audio_target: torch.Tensor        # (B, N_audio, 32)
    layout: PackedLayout
    video_t: torch.Tensor             # (B,) for logging / loss weighting
    audio_t: torch.Tensor

    def transformer_kwargs(self) -> dict:
        return dict(
            hidden_states=self.video_rows,
            audio_hidden_states=self.audio_rows,
            encoder_hidden_states=self.text_embeds,
            timestep=self.timestep,
            timestep_indices=self.timestep_indices,
            token_tags=self.layout.token_tags,
            position_ids=self.layout.position_ids,
            video_indices=self.layout.video_indices,
            audio_indices=self.layout.audio_indices,
            text_indices=self.layout.text_indices,
            return_dict=False,
        )


def make_flow_batch(video_latents: torch.Tensor, audio_latents: torch.Tensor,
                    text_embeds: torch.Tensor, layout: PackedLayout,
                    generator=None, timestep_mode: str = "uniform",
                    condition_rows: torch.Tensor | None = None,
                    video_shift: float = VIDEO_FLOW_SHIFT,
                    audio_shift: float = AUDIO_FLOW_SHIFT,
                    fixed_t: tuple[float, float] | None = None) -> FlowBatch:
    """Noise one batch and assemble the transformer's arguments.

    Args:
        video_latents: `(B, 24, F, H, W)` clean video latents from the frozen VAE.
        audio_latents: `(B, 2, 32, L)` clean audio latents, per channel.
        text_embeds:   `(B, N_text, 5120)` Qwen3-VL layer-50 hidden states.
        layout:        the shared packed layout.
        condition_rows: `(B, n_cond, 96)` already-noised keyframe anchor rows, or
            None for the text-only task. Anchors are held at `KEYFRAME_NOISE_AUG`
            and are prepended to the video rows, matching the packed order.

    **One timestep pair per batch, not per item.** `timestep_indices` is a
    `(seq_len,)` tensor, not `(batch, seq_len)`: the transformer's batch axis is a
    pure replication axis and the architecture cannot express a different noise level
    per item. Drawing per item and noising each at its own `t` while the AdaLN table
    states item 0's would train the model against a timestep its input does not have,
    which teaches it to ignore `t`. The two modalities still draw *independently* of
    each other, which is the part that matters: video and audio each get their own
    shifted schedule inside the same forward pass.
    """
    device = video_latents.device
    batch = video_latents.shape[0]

    if fixed_t is not None:
        # Held-out evaluation pins the timesteps so two checkpoints are scored on the
        # same noise levels; a fresh draw per checkpoint would put most of the
        # variance between runs into the draw rather than into the models.
        video_t = torch.full((batch,), float(fixed_t[0]), device=device)
        audio_t = torch.full((batch,), float(fixed_t[1]), device=device)
    else:
        video_t = sample_timesteps(1, video_shift, device, generator, timestep_mode).expand(batch)
        audio_t = sample_timesteps(1, audio_shift, device, generator, timestep_mode).expand(batch)

    video_rows = patchify_video_latents(video_latents)                 # (B, Nv_gen, 96)
    audio_rows = pack_audio_channel_major(audio_latents)               # (B, Na, 32)

    video_noise = torch.randn(video_rows.shape, device=device, dtype=video_rows.dtype, generator=generator)
    audio_noise = torch.randn(audio_rows.shape, device=device, dtype=audio_rows.dtype, generator=generator)

    noisy_video = add_noise(video_rows, video_noise, video_t)
    noisy_audio = add_noise(audio_rows, audio_noise, audio_t)

    video_target = velocity_target(video_rows, video_noise)
    audio_target = velocity_target(audio_rows, audio_noise)

    if condition_rows is not None:
        # Conditioning rows lead the video rows and are never a loss term; a zero
        # target keeps the tensors aligned and `loss_mask` drops them.
        noisy_video = torch.cat([condition_rows, noisy_video], dim=1)
        video_target = torch.cat([torch.zeros_like(condition_rows), video_target], dim=1)

    # Guard against a regression to per-item timesteps: the row-timestep table below
    # can only state one value per modality, so a batch whose items were noised at
    # different levels would be trained against a timestep it does not have.
    if batch > 1 and not (bool(video_t.eq(video_t[0]).all()) and bool(audio_t.eq(audio_t[0]).all())):
        raise ValueError(
            "MiniMax-H3 shares one `timestep_indices` across the batch, so every item "
            "must be noised at the same (video, audio) timestep pair."
        )
    timestep, timestep_indices = build_row_timesteps(
        layout, float(video_t[0]), float(audio_t[0])
    )
    return FlowBatch(
        video_rows=noisy_video, audio_rows=noisy_audio, text_embeds=text_embeds,
        timestep=timestep.to(device), timestep_indices=timestep_indices.to(device),
        video_target=video_target, audio_target=audio_target, layout=layout,
        video_t=video_t, audio_t=audio_t,
    )


def flow_loss(video_pred: torch.Tensor, audio_pred: torch.Tensor, batch: FlowBatch,
              audio_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean-squared velocity error over the *generated* rows of both modalities.

    Conditioning rows are dropped rather than down-weighted: the transformer emits a
    velocity for every row it is given, including the anchors, and nothing in the
    sampler ever consumes those, so training against them would spend capacity on an
    output that is thrown away.

    `audio_weight` is exposed because the two modalities contribute wildly different
    row counts -- at 256x256 and two seconds it is ~832 video rows against ~160 audio
    rows -- so an unweighted mean over rows is already a 5:1 bias toward video.
    """
    n_cond = batch.layout.num_condition_video_rows
    video_pred = video_pred[:, n_cond:]
    video_true = batch.video_target[:, n_cond:]
    n_cond_a = batch.layout.num_condition_audio_rows
    audio_pred = audio_pred[:, n_cond_a:]
    audio_true = batch.audio_target[:, n_cond_a:]

    video_mse = (video_pred.float() - video_true.float()).pow(2).mean()
    audio_mse = (audio_pred.float() - audio_true.float()).pow(2).mean()
    total = video_mse + audio_weight * audio_mse
    return total, {
        "loss": float(total.detach()),
        "loss_video": float(video_mse.detach()),
        "loss_audio": float(audio_mse.detach()),
        "t_video": float(batch.video_t.mean()),
        "t_audio": float(batch.audio_t.mean()),
    }
