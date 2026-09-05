"""Known-answer tests for the failure-pool reliability measurements.

Each test fixes a case whose answer is known before the code runs, so a wrong
implementation fails rather than producing a plausible number. The one that
earns its place is ``test_cross_null_separates_conditional_from_marginal``:
the first version of the null in this analysis applied a single permutation to
both halves, which leaves the mis-pairing shared and therefore agreed-upon, and
reported reliability 0.49 where the answer should have been 0. That test fails
on that mistake.
"""

import numpy as np
import pytest

import failure_pool_reliability as F


# --- the decomposition ----------------------------------------------------

def test_total_partitions_exactly_into_bias_and_dispersion():
    rng = np.random.default_rng(0)
    real, gen = rng.normal(size=(40, 7)), rng.normal(size=(9, 40, 7))
    s = F.scores(real, gen)
    assert np.allclose(s["total"], s["bias2_raw"] + s["spread_pop"], atol=1e-12)


def test_bias_correction_is_unbiased_where_the_truth_is_known():
    """(xbar - y)^2 over-states bias^2 by sigma^2/k; the correction removes it."""
    rng = np.random.default_rng(1)
    bias, sigma, k, trials = 0.5, 2.0, 4, 40_000
    x = rng.normal(bias, sigma, size=(k, trials, 1))
    real = np.zeros((trials, 1))
    s = F.scores(real, x)
    assert s["bias2_raw"].mean() == pytest.approx(bias**2 + sigma**2 / k, rel=0.05)
    assert s["bias2"].mean() == pytest.approx(bias**2, abs=0.05)


def test_single_rollout_cannot_separate_the_two_terms():
    rng = np.random.default_rng(2)
    s = F.scores(rng.normal(size=(20, 3)), rng.normal(size=(1, 20, 3)))
    assert np.allclose(s["spread_pop"], 0.0)
    assert np.allclose(s["total"], s["bias2_raw"])


# --- rank machinery -------------------------------------------------------

def test_ties_are_averaged_not_ordered_by_position():
    r = F._rank(np.array([1.0, 1.0, 1.0, 2.0, 3.0]))
    assert list(r) == [1.0, 1.0, 1.0, 3.0, 4.0]


def test_spearman_endpoints():
    x = np.arange(50.0)
    assert F.spearman(x, x) == pytest.approx(1.0)
    assert F.spearman(x, -x) == pytest.approx(-1.0)


def test_a_constant_score_correlates_with_nothing():
    """All-ties must give nan, not 1.0 from array order."""
    rng = np.random.default_rng(3)
    assert np.isnan(F.spearman(np.ones(50), rng.normal(size=50)))


# --- the correction -------------------------------------------------------

def test_stratify_removes_a_score_that_is_only_the_realised_move():
    """A score that is a monotone function of |y| carries no context signal.

    Raw, it correlates perfectly with |y|. Stratified, it must fall to noise:
    inside a bin of nearly equal |y| there is nothing left to rank.
    """
    rng = np.random.default_rng(4)
    y = rng.normal(size=4000)
    score = np.abs(y) ** 2
    assert F.spearman(score, np.abs(y)) == pytest.approx(1.0, abs=1e-9)
    assert abs(F.spearman(F.stratify(score, y, n_bins=20), np.abs(y))) < 0.10


def test_stratify_keeps_a_signal_orthogonal_to_the_realised_move():
    rng = np.random.default_rng(5)
    y = rng.normal(size=4000)
    signal = rng.normal(size=4000)
    score = np.abs(y) ** 2 + 0.5 * signal
    assert F.spearman(F.stratify(score, y, n_bins=20), signal) > 0.5


# --- the nulls ------------------------------------------------------------

def _synthetic(n_ctx, n_seed, rng, context_signal):
    """Rollouts whose error either does or does not depend on the context.

    context_signal=0 makes every context statistically identical, so no pairing
    can matter. context_signal>0 gives each context its own systematic offset,
    which only the correct pairing can see.
    """
    real = np.zeros((n_ctx, 1))
    offs = context_signal * rng.normal(size=(1, n_ctx, 1))
    return real, offs + rng.normal(size=(n_seed, n_ctx, 1))


def test_cross_null_separates_conditional_from_marginal():
    """The test that catches a shared-permutation null.

    With a real per-context signal: `true` must beat `independent`, and `cross`
    (true half vs permuted half) must fall back to `independent`, because
    scoring against the wrong context destroys the very thing being ranked.
    `shared` is deliberately not asserted low --- a consistently wrong pairing
    is still self-consistent, which is exactly why reliability alone proves
    nothing.
    """
    rng = np.random.default_rng(6)
    real, gen = _synthetic(600, 12, rng, context_signal=1.5)
    n = F.pairing_nulls(real, gen, k=6, horizon_idx=0, rng=rng, stratified=False)
    assert n["true"] > 0.5
    assert abs(n["independent"]) < 0.15
    assert abs(n["cross"]) < 0.15
    assert n["true"] - n["cross"] > 0.35


def test_all_nulls_vanish_when_no_context_signal_exists():
    rng = np.random.default_rng(7)
    real, gen = _synthetic(600, 12, rng, context_signal=0.0)
    n = F.pairing_nulls(real, gen, k=6, horizon_idx=0, rng=rng, stratified=False)
    for key in ("true", "shared", "independent", "cross"):
        assert abs(n[key]) < 0.15, f"{key} = {n[key]}"


# --- extrapolation --------------------------------------------------------

def test_rollouts_needed_recovers_a_planted_noise_to_signal_ratio():
    ratio = 3.0
    ks = [1, 2, 3, 5, 8]
    rhos = [1.0 / (1.0 + ratio / k) for k in ks]
    out = F.rollouts_needed(ks, rhos)
    assert out["noise_over_signal"] == pytest.approx(ratio, rel=1e-6)
    assert out["max_abs_resid"] < 1e-9
    assert out["k_for_rho_0.80"] == pytest.approx(ratio / 0.25, rel=1e-6)


def test_pool_overlap_is_total_when_the_correction_changes_nothing():
    """If |y| is constant there are no strata to re-rank within, so the two
    pools must coincide."""
    rng = np.random.default_rng(8)
    real = np.full((300, 1), 0.01)
    gen = rng.normal(size=(6, 300, 1))
    assert F.pool_overlap(real, gen, horizon_idx=0) == pytest.approx(1.0)


# --- guards added 2026-09-05 after an adversarial audit of commit e8425cb1 -------------
#
# Each of these goes red on a defect that was actually shipped, not on a hypothetical one.
# The audit is at
# /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_20260904/drafts/D5_premortem.md

def test_stratification_leak_is_exactly_one_over_n_bins():
    """The published "zero line" of 0.095 was this leak, not noise.

    A score that is a pure monotone function of |y| still correlates 1/n_bins with |y|
    after stratification: within-bin ranking cannot undo the between-bin ordering. The
    original leak test ran at n_bins=20 (leak 0.05) while production ran at n_bins=10
    (leak 0.10), so it passed at twice the production tolerance and the floor went
    unnoticed. Pin the identity itself.
    """
    rng = np.random.default_rng(4)
    y = rng.normal(size=40_000)
    score = np.abs(y) ** 2
    for n_bins in (5, 10, 20, 40):
        got = abs(F.spearman(F.stratify(score, y, n_bins=n_bins), np.abs(y)))
        assert got == pytest.approx(F.stratification_leak(n_bins), abs=0.005), n_bins


def test_stratify_at_production_bins_leaves_the_leak_not_zero():
    """At the parameter production actually uses, the floor is 0.10, not 0."""
    rng = np.random.default_rng(5)
    y = rng.normal(size=40_000)
    got = abs(F.spearman(F.stratify(np.abs(y) ** 2, y, n_bins=10), np.abs(y)))
    assert got > 0.05, "a leak this small would mean the identity above is wrong"
    assert got == pytest.approx(0.10, abs=0.005)


def test_rollouts_needed_refuses_to_extrapolate_a_curve_that_is_not_linear():
    """The planted-linear test could never fail on real data.

    The implied noise/signal ratio (1/rho_k - 1)*k rises with k in 7 of 8 tickers, so the
    one-parameter law is rejected by the data it was fitted to, and least squares through
    the origin on x=1/k is dominated by k=1 and extrapolates k too low. A point estimate
    the data rejects is worse than none, because it gets quoted.
    """
    ks = [1, 2, 3, 5]
    rising = [1.0 / (1.0 + r / k) for k, r in zip(ks, [4.68, 6.79, 8.37, 9.87])]  # GOOG
    out = F.rollouts_needed(ks, rising)
    assert out["ratio_rises_with_k"] is True
    assert "rejected_reason" in out
    assert "k_for_rho_0.80" not in out, "emitted a point estimate its own residual rejects"
    assert out["k_for_rho_0.80_largest_k_only"] > out["noise_over_signal"] / (1 / 0.80 - 1)


def test_rollouts_needed_still_answers_when_the_law_does_hold():
    ratio, ks = 3.0, [1, 2, 3, 5, 8]
    out = F.rollouts_needed(ks, [1.0 / (1.0 + ratio / k) for k in ks])
    assert "rejected_reason" not in out
    assert out["k_for_rho_0.80"] == pytest.approx(ratio / 0.25, rel=1e-6)


def test_dispersion_share_reports_the_unbiased_value():
    """spread_pop is ddof=0 and understates sigma^2 by (k-1)/k while `total` is unbiased,
    so the published share understated the irreducible part by ~11% at k=10."""
    rng = np.random.default_rng(6)
    real = rng.normal(size=(400, 1))
    gen = real[None] + rng.normal(size=(10, 400, 1))
    d = F.dispersion_share(real, gen)["per_horizon"][0]
    assert d["spread_share_top_decile_unbiased"] == pytest.approx(
        d["spread_share_top_decile"] * 10 / 9, rel=1e-9)
    assert d["max_removable_share_top_decile"] == pytest.approx(
        1.0 - d["spread_share_top_decile_unbiased"], rel=1e-9)


# --- guards for the repeated nulls and the discriminating null (items 2 and 3) ---------

def test_repeated_nulls_report_a_spread_not_a_single_draw():
    """One draw cannot order `shared` against `true`; the published 0.49-vs-0.46 was one."""
    rng = np.random.default_rng(20)
    real = np.zeros((400, 1))
    gen = 1.5 * rng.normal(size=(1, 400, 1)) + rng.normal(size=(9, 400, 1))
    out = F.pairing_nulls_repeated(real, gen, k=3, horizon_idx=0, rng=rng,
                                   stratified=False, n_draws=15)
    assert out["n_draws"] == 15
    for key in ("true", "shared", "independent", "cross"):
        assert out[f"{key}_sd"] > 0, f"{key} has no spread, so it is still one draw"
    assert 0.0 <= out["shared_exceeds_true_frac"] <= 1.0


def test_repeated_nulls_carry_the_leak_floor_when_stratified():
    """A stratified null is bounded below by 1/n_bins; the reading must say so."""
    rng = np.random.default_rng(21)
    real, gen = np.zeros((300, 1)), rng.normal(size=(9, 300, 1))
    assert F.pairing_nulls_repeated(real, gen, 3, 0, rng, True, 5)["leak_floor"] == \
        pytest.approx(F.stratification_leak())
    assert F.pairing_nulls_repeated(real, gen, 3, 0, rng, False, 5)["leak_floor"] == 0.0


def test_partialling_mostly_removes_a_score_that_is_only_dispersion():
    """Known-answer case, and it establishes that the floor is not zero.

    Rollouts are centred on the truth and differ only in width, so `total` is pure
    dispersion and carries nothing about being wrong. The honest answer would be 0 kept.
    It is not: the nuisance is estimated from held-out seeds, and partialling a noisy
    proxy under-removes. The measured residue (~0.37 at k=3 with 6 held-out seeds) is the
    floor that the real-data figure of ~0.87 must be quoted against -- the same class of
    error as reading a stratified null of 0.10 as "zero".
    """
    rng = np.random.default_rng(22)
    floor = F.dispersion_partial_floor(n_ctx=500, n_seed=12, k=3, rng=rng, n_draws=20)
    assert 0.15 < floor < 0.55, f"floor {floor:.2f} outside the range this control produces"
    # most of it does go, which is what makes the partial worth doing at all
    assert floor < 0.6


def test_partial_out_removes_a_planted_monotone_nuisance():
    """A monotone function of the nuisance has the same ranks, so the residual is exactly
    zero -- and `spearman` on a constant is NaN by design, so assert on the residual."""
    rng = np.random.default_rng(23)
    nuis = rng.normal(size=800)
    assert np.abs(F.partial_out(np.exp(nuis), nuis)).max() < 1e-9
    # a nuisance that explains only part of the target leaves a real residual
    partial = F.partial_out(np.exp(nuis) + 3.0 * rng.normal(size=800), nuis)
    assert abs(F.spearman(partial, nuis)) < 0.10
