#!/usr/bin/env python
"""Executable checks on the H3 conventions H3-nano is built against.

Runs on CPU in seconds with tiny tensors. The point is not coverage, it is that
every assertion here corresponds to something that can be silently wrong: a sign
flipped in the flow-matching convention, an axis transposed in the audio packing,
a timestep drawn per item when the architecture cannot express one. None of those
raise on their own, and none of them show up in a loss curve.

    ./venv/bin/python code/test_h3nano.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3nano as H  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ---------------------------------------------------------------- conventions
@check
def velocity_target_is_data_minus_noise():
    """`v = x0 - eps`, and it must not depend on `t`.

    Derived from the scheduler's `x0 = x_t + sigma * v`. Recomputing it at several
    timesteps catches a version that accidentally carries a `t` factor.
    """
    x0 = torch.randn(4, 16)
    eps = torch.randn(4, 16)
    target = H.velocity_target(x0, eps)
    assert torch.allclose(target, x0 - eps)
    for t_value in (0.05, 0.4, 0.95):
        t = torch.full((4,), t_value)
        x_t = H.add_noise(x0, eps, t)
        recovered = (x0 - x_t) / (1.0 - t_value)
        assert torch.allclose(recovered, target, atol=1e-5), f"t={t_value}"
    return "v = x0 - eps at t in {0.05, 0.4, 0.95}"


@check
def denoising_a_perfect_velocity_lands_on_x0():
    """The strongest end-to-end check of the sign convention.

    A model that predicted the exact velocity should, stepped through the *real*
    `MiniMaxH3Scheduler` from pure noise, arrive at `x0`. Any flipped sign in
    `v = x0 - eps`, in `t = 1 - sigma`, or in the data-ward `x0 = x_t + sigma * v`
    sends the trajectory somewhere else, so this exercises the whole convention
    against the reference implementation rather than against itself.
    """
    from diffusers import MiniMaxH3Scheduler

    torch.manual_seed(0)
    x0 = torch.randn(2, 8)
    eps = torch.randn(2, 8)
    scheduler = MiniMaxH3Scheduler(shift=H.VIDEO_FLOW_SHIFT)
    scheduler.set_timesteps(64)

    sample = eps.clone()                    # sigma = 1 is pure noise, t = 0
    for t in scheduler.timesteps:
        v = H.velocity_target(x0, eps)      # the oracle velocity, constant in t
        sample = scheduler.step(v, t, sample, return_dict=False)[0]
    error = (sample - x0).abs().max()
    assert error < 1e-3, f"perfect-velocity denoising ended {error:.3e} from x0"
    return f"64 steps from pure noise land within {float(error):.2e} of x0"


@check
def the_two_shifts_produce_different_grids():
    """`shift = 12` (video) must spend far more of its steps at high sigma than 3 (audio)."""
    from diffusers import MiniMaxH3Scheduler

    grids = {}
    for name, shift in (("video", H.VIDEO_FLOW_SHIFT), ("audio", H.AUDIO_FLOW_SHIFT)):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(33)
        grids[name] = scheduler.sigmas
    video_high = float((grids["video"] > 0.5).float().mean())
    audio_high = float((grids["audio"] > 0.5).float().mean())
    assert video_high > audio_high, "video should concentrate more mass at high sigma"
    return f"fraction of grid above sigma=0.5: video {video_high:.2f} vs audio {audio_high:.2f}"


# ---------------------------------------------------------------- packing
@check
def patchify_round_trips_exactly():
    latents = torch.randn(3, 24, 7, 8, 8)
    rows = H.patchify_video_latents(latents)
    assert rows.shape == (3, 7 * 4 * 4, 96), rows.shape
    back = H.unpatchify_video_latents(rows, 7, 8, 8)
    assert torch.equal(back, latents)
    return f"(3, 24, 7, 8, 8) <-> {tuple(rows.shape)} exact"


@check
def audio_packing_is_channel_major_and_exact():
    """Row order must be [ch0 over time, then ch1 over time].

    `_fill_audio_positions` writes the rotary time as `arange(L).repeat(channels)` and
    pins the width coordinate to the first grid value for the first `L` rows, so a
    time-major packing would put the right channel's samples at the left channel's
    position and silently swap the stereo image.
    """
    latents = torch.arange(2 * 32 * 5, dtype=torch.float32).reshape(1, 2, 32, 5)
    rows = H.pack_audio_channel_major(latents)
    assert rows.shape == (1, 10, 32)
    assert torch.equal(rows[0, 0], latents[0, 0, :, 0]), "first row must be channel 0, time 0"
    assert torch.equal(rows[0, 5], latents[0, 1, :, 0]), "row L must start channel 1"
    assert torch.equal(H.unpack_audio_channel_major(rows), latents)
    return "channel-major order confirmed and exact on round trip"


# ---------------------------------------------------------------- geometry
@check
def frame_geometry_matches_the_reference():
    for frames in (1, 5, 20, 60, 73, 124, 141, 345):
        snapped = H.snap_num_frames(frames)
        assert snapped % 17 == 5, f"{frames} -> {snapped}"
        assert snapped >= frames or frames < 5
        latents = H.num_latent_frames_for(snapped)
        assert latents == 5 * ((snapped - 5) // 17) + 2
    audio = H.num_audio_latents_for(124)
    assert audio == round(124 / 24 * 40) == 207
    return "17n+5 -> 5n+2 and 40 Hz audio confirmed over 8 frame counts"


@check
def layout_row_counts_follow_the_geometry():
    text, frames, h, w, audio = 96, 22, 16, 16, 122
    layout = H.build_layout(text, frames, h, w, audio)
    rows_per_frame = (h // 2) * (w // 2)
    assert layout.rows_per_frame == rows_per_frame
    assert len(layout.text_indices) == text
    assert len(layout.audio_indices) == audio * H.AUDIO_CHANNELS
    assert len(layout.video_indices) == frames * rows_per_frame
    assert layout.sequence_length == text + audio * 2 + frames * rows_per_frame
    tags = torch.bincount(layout.token_tags, minlength=3)
    assert tags[H.TEXT_TAG] == text and tags[H.AUDIO_TAG] == audio * 2
    return f"seq={layout.sequence_length} = text {text} + audio {audio*2} + video {frames*rows_per_frame}"


@check
def keyframe_anchors_add_leading_conditioning_rows():
    text, frames, h, w, audio = 16, 7, 8, 8, 37
    plain = H.build_layout(text, frames, h, w, audio)
    anchored = H.build_layout(text, frames, h, w, audio, keyframe_anchors=("first", "last"))
    extra = anchored.sequence_length - plain.sequence_length
    assert extra == 2 * plain.rows_per_frame, extra
    assert anchored.num_condition_video_rows == 2 * plain.rows_per_frame
    return f"two anchors add {extra} leading video rows"


# ---------------------------------------------------------------- timesteps
@check
def text_rows_inherit_the_video_timestep():
    """A row-timestep table that gives text its own level would be wrong.

    The reference fills the whole sequence with the video timestep and only then
    overwrites audio and conditioning rows, so text tracks video. That makes the text
    conditioning move along the denoising trajectory rather than staying static.
    """
    layout = H.build_layout(8, 7, 8, 8, 37)
    timestep, indices = H.build_row_timesteps(layout, video_t=0.3, audio_t=0.7)
    text_levels = timestep[indices[layout.text_indices]]
    video_levels = timestep[indices[layout.video_indices]]
    audio_levels = timestep[indices[layout.audio_indices]]
    assert torch.allclose(text_levels, torch.full_like(text_levels, 0.3))
    assert torch.allclose(video_levels, torch.full_like(video_levels, 0.3))
    assert torch.allclose(audio_levels, torch.full_like(audio_levels, 0.7))
    assert len(timestep) == 2, "two distinct levels should collapse to two table rows"
    return "text follows video (0.3); audio independent (0.7); table has 2 rows"


@check
def anchors_sit_at_the_documented_noise_level():
    layout = H.build_layout(8, 7, 8, 8, 37, keyframe_anchors=("first",))
    timestep, indices = H.build_row_timesteps(layout, video_t=0.3, audio_t=0.7)
    n_cond = layout.num_condition_video_rows
    anchor_levels = timestep[indices[layout.video_indices[:n_cond]]]
    assert torch.allclose(anchor_levels, torch.full_like(anchor_levels, H.KEYFRAME_NOISE_AUG))
    return f"anchor rows held at max(t, {H.KEYFRAME_NOISE_AUG})"


@check
def a_batch_shares_one_timestep_pair():
    """The architecture cannot express per-item timesteps, so the guard must fire."""
    layout = H.build_layout(8, 7, 8, 8, 37)
    video = torch.randn(3, 24, 7, 8, 8)
    audio = torch.randn(3, 2, 32, 37)
    text = torch.randn(3, 8, 5120)
    batch = H.make_flow_batch(video, audio, text, layout)
    assert torch.allclose(batch.video_t, batch.video_t[0].expand_as(batch.video_t))
    assert len(batch.timestep) == 2, "video and audio draw independently"

    original = H.sample_timesteps
    H.sample_timesteps = lambda n, s, d, g=None, m="uniform": torch.rand(3, device=d)
    try:
        H.make_flow_batch(video, audio, text, layout)
    except ValueError:
        fired = True
    else:
        fired = False
    finally:
        H.sample_timesteps = original
    assert fired, "per-item timesteps must raise, not be silently accepted"
    return "batch shares one (t_video, t_audio); per-item draws raise"


# ---------------------------------------------------------------- loss
@check
def loss_ignores_conditioning_rows():
    """A wrong prediction on an anchor row must not move the loss.

    The transformer emits a velocity for every row it is given, including anchors,
    and the sampler never consumes those, so training against them spends capacity
    on a discarded output.
    """
    layout = H.build_layout(8, 7, 8, 8, 37, keyframe_anchors=("first",))
    n_cond = layout.num_condition_video_rows
    video = torch.randn(2, 24, 7, 8, 8)
    audio = torch.randn(2, 2, 32, 37)
    text = torch.randn(2, 8, 5120)
    condition = torch.randn(2, n_cond, 96)
    batch = H.make_flow_batch(video, audio, text, layout, condition_rows=condition)

    video_pred = batch.video_target.clone()
    audio_pred = batch.audio_target.clone()
    clean, _ = H.flow_loss(video_pred, audio_pred, batch)
    video_pred[:, :n_cond] += 1000.0                       # ruin only the anchors
    polluted, _ = H.flow_loss(video_pred, audio_pred, batch)
    assert torch.allclose(clean, polluted), f"{float(clean)} vs {float(polluted)}"
    assert float(clean) < 1e-10
    return f"perturbing {n_cond} anchor rows by 1000 leaves the loss unchanged"


@check
def model_forward_shapes_match_the_layout():
    layout = H.build_layout(8, 7, 8, 8, 37)
    model = H.build_transformer(H.MICRO_CONFIG)
    video = torch.randn(2, 24, 7, 8, 8)
    audio = torch.randn(2, 2, 32, 37)
    text = torch.randn(2, 8, 5120)
    batch = H.make_flow_batch(video, audio, text, layout)
    with torch.no_grad():
        video_pred, audio_pred = model(**batch.transformer_kwargs())
    assert video_pred.shape == batch.video_target.shape
    assert audio_pred.shape == batch.audio_target.shape
    assert torch.isfinite(video_pred).all() and torch.isfinite(audio_pred).all()
    census = H.parameter_census(model)
    assert census["token_refiner"] > 1e6, "refiner must not be swallowed by the attn/ff buckets"
    return f"micro forward OK; {census['TOTAL']/1e6:.2f} M params, refiner {census['token_refiner']/1e6:.2f} M"


def main() -> int:
    torch.manual_seed(0)
    failures = 0
    width = max(len(fn.__name__) for fn in CHECKS)
    for fn in CHECKS:
        try:
            detail = fn()
            print(f"  PASS  {fn.__name__:<{width}}  {detail}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__:<{width}}  {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
