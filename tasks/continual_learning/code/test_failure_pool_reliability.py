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
