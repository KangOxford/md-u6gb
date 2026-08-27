"""CPU unit tests for plasticity_probes (PLAN.md Step 1).

Run with ``pytest test_plasticity_probes.py`` or plain
``python test_plasticity_probes.py``.
"""

import numpy as np

from plasticity_probes import (
    OptimizationReadiness,
    dormant_fraction,
    effective_rank,
    global_l2_norm,
    top_hessian_eigenvalue,
)


def test_dormant_half_zero():
    rng = np.random.default_rng(0)
    acts = rng.standard_normal((512, 64))
    acts[:, 32:] = 0.0
    assert dormant_fraction(acts) == 0.5


def test_dormant_all_zero_layer():
    assert dormant_fraction(np.zeros((128, 16))) == 1.0


def test_dormant_uniform_layer_has_none():
    rng = np.random.default_rng(1)
    acts = rng.standard_normal((2048, 32))
    assert dormant_fraction(acts) == 0.0


def test_effective_rank_isotropic():
    rng = np.random.default_rng(2)
    feats = rng.standard_normal((20000, 16))
    er = effective_rank(feats)
    assert abs(er - 16.0) < 0.5


def test_effective_rank_rank_one():
    rng = np.random.default_rng(3)
    feats = np.outer(rng.standard_normal(1000), rng.standard_normal(24))
    er = effective_rank(feats, center=False)
    assert abs(er - 1.0) < 1e-9


def test_effective_rank_degenerate_is_zero():
    assert effective_rank(np.zeros((100, 8))) == 0.0
    # constant features center to zero
    assert effective_rank(np.ones((100, 8))) == 0.0


def test_global_l2_norm():
    assert abs(global_l2_norm([np.array([3.0]), np.array([4.0])]) - 5.0) < 1e-12


def test_readiness_identical_gradients():
    g = np.arange(1.0, 6.0)  # ||g||^2 = 55
    acc = OptimizationReadiness()
    for _ in range(4):
        acc.update(g)
    assert abs(acc.reliability - 1.0) < 1e-12
    assert abs(acc.strength - 55.0) < 1e-12
    assert abs(acc.readiness - 55.0) < 1e-12


def test_readiness_cancelling_gradients():
    g = np.ones(7)
    acc = OptimizationReadiness()
    acc.update(g)
    acc.update(-g)
    assert acc.strength < 1e-24
    assert acc.readiness < 1e-24


def test_readiness_matches_direct_formula():
    rng = np.random.default_rng(4)
    grads = rng.standard_normal((6, 40))
    acc = OptimizationReadiness()
    for g in grads:
        acc.update(g)
    gbar = grads.mean(axis=0)
    strength = float(gbar @ gbar)
    mean_sq = float(np.square(grads).sum(axis=1).mean())
    assert abs(acc.strength - strength) < 1e-10
    assert abs(acc.reliability - strength / mean_sq) < 1e-10
    assert abs(acc.readiness - strength * strength / mean_sq) < 1e-10


def test_readiness_list_of_leaves_matches_flat():
    rng = np.random.default_rng(5)
    grads = rng.standard_normal((3, 30))
    flat = OptimizationReadiness()
    tree = OptimizationReadiness()
    for g in grads:
        flat.update(g)
        tree.update([g[:11].reshape(11), g[11:].reshape(19)])
    assert abs(flat.readiness - tree.readiness) < 1e-12


def test_top_hessian_eigenvalue_positive_dominant():
    rng = np.random.default_rng(6)
    q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    eigs = np.linspace(-3.0, 9.0, 32)
    a = q @ np.diag(eigs) @ q.T
    val, iters = top_hessian_eigenvalue(lambda v: a @ v, dim=32, n_iters=500, tol=1e-12)
    assert abs(val - 9.0) < 1e-6
    assert iters <= 500


def test_top_hessian_eigenvalue_negative_dominant():
    a = np.diag([-8.0, 2.0, 1.0])
    val, _ = top_hessian_eigenvalue(lambda v: a @ v, dim=3, n_iters=500, tol=1e-14)
    assert abs(val - (-8.0)) < 1e-6


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
