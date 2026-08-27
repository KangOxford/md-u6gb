"""Plasticity probes: framework-agnostic diagnostics for continual learning.

Implements the five probes of PLAN.md Step 1 so that every long training run
can log plasticity evidence by default:

- dormant_fraction        ReDo-style dormant-unit fraction (arXiv 2302.12902)
- effective_rank          Renyi-2 effective rank of the feature covariance
- global_l2_norm          global L2 norm over a collection of arrays
- OptimizationReadiness   streaming gradient strength x reliability score,
                          reconstructed from the descriptor of arXiv 2605.09044
- top_hessian_eigenvalue  power iteration over an injected Hessian-vector
                          product callable (Pearlmutter trick lives caller-side)

Everything takes plain numpy arrays. JAX callers pass
``np.asarray(jax.device_get(x))``; PyTorch callers pass
``t.detach().cpu().numpy()``. The module never imports a training framework,
so it is testable on a CPU-only login node.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Union

import numpy as np

Array = np.ndarray
Arrays = Union[Array, Sequence[Array]]

__all__ = [
    "dormant_fraction",
    "effective_rank",
    "global_l2_norm",
    "OptimizationReadiness",
    "top_hessian_eigenvalue",
]


def _as_array_list(x: Arrays) -> List[Array]:
    if isinstance(x, np.ndarray):
        return [x]
    return [np.asarray(a) for a in x]


def dormant_fraction(activations: Array, eps: float = 0.01) -> float:
    """Fraction of dormant units in one layer, ReDo-style.

    A unit's score is its mean absolute activation over samples, normalized by
    the layer-mean score; a unit is dormant when the normalized score is at
    most ``eps``. A layer whose activations are identically zero is fully
    dormant (fraction 1.0).

    Args:
        activations: array of shape (n_samples, n_units) for one layer
                     (flatten any leading batch/sequence dims into axis 0).
        eps: dormancy threshold on the normalized score (default 0.01,
             matching the ReDo definition used by the Zyphra study).
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"expected (n_samples, n_units), got shape {acts.shape}")
    score = np.abs(acts).mean(axis=0)
    layer_mean = score.mean()
    if layer_mean <= 0.0:
        return 1.0
    return float((score / layer_mean <= eps).mean())


def effective_rank(features: Array, center: bool = True) -> float:
    """Renyi-2 effective rank er2(M) = (tr M)^2 / ||M||_F^2 of the feature
    covariance M, computed from singular values without forming M.

    Ranges from 1 (rank-one covariance) to d (isotropic covariance); returns
    0.0 for identically-zero (or constant, when centered) features. Falling
    er2 across training is one of the pre-registered plasticity diagnostics.

    Args:
        features: array of shape (n_samples, d).
        center: subtract the feature mean first (covariance rather than
                second-moment matrix).
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"expected (n_samples, d), got shape {feats.shape}")
    if center:
        feats = feats - feats.mean(axis=0, keepdims=True)
    # Eigenvalues of M = X^T X / n are s_i^2 / n; er2 is scale-invariant, so
    # the 1/n factor cancels and the singular values of X suffice.
    s = np.linalg.svd(feats, compute_uv=False)
    lam = s * s
    total = lam.sum()
    if total <= 0.0:
        return 0.0
    return float(total * total / np.square(lam).sum())


def global_l2_norm(arrays: Arrays) -> float:
    """Global L2 norm over a collection of arrays (e.g. a parameter or
    gradient tree already flattened to leaves). Callers decide which leaves
    to include (PLAN.md: exclude embeddings for the weight-norm probe)."""
    total = 0.0
    for a in _as_array_list(arrays):
        a = np.asarray(a, dtype=np.float64)
        total += float(np.square(a).sum())
    return float(np.sqrt(total))


class OptimizationReadiness:
    """Streaming gradient strength x reliability score.

    Reconstruction of the "optimization readiness" descriptor of Wang et al.
    (arXiv 2605.09044): combine gradient strength with gradient reliability so
    that the product lower-bounds the expected one-step optimization gain. With
    micro-batch gradients g_1..g_K and gbar = mean g_i, a step of optimally
    sized SGD gains at least ||gbar||^4 / (2 beta mean ||g_i||^2) under
    beta-smoothness, which factors as

        strength    = ||gbar||^2
        reliability = ||gbar||^2 / mean ||g_i||^2   (in [0, 1])
        readiness   = strength * reliability

    The exact estimator of the paper may differ in normalization; trends, not
    absolute values, are what the plasticity dashboard consumes.

    Memory: keeps one gradient-shaped accumulator plus two scalars, so K
    micro-batch gradients never need to be resident together.
    """

    def __init__(self) -> None:
        self._sum: List[Array] | None = None
        self._sum_sq_norms = 0.0
        self._count = 0

    def update(self, grad: Arrays) -> None:
        """Accumulate one micro-batch gradient (array or list of leaves)."""
        leaves = [np.asarray(a, dtype=np.float64) for a in _as_array_list(grad)]
        if self._sum is None:
            self._sum = [np.zeros_like(a) for a in leaves]
        if len(leaves) != len(self._sum) or any(
            a.shape != b.shape for a, b in zip(leaves, self._sum)
        ):
            raise ValueError("gradient structure changed between updates")
        sq = 0.0
        for acc, a in zip(self._sum, leaves):
            acc += a
            sq += float(np.square(a).sum())
        self._sum_sq_norms += sq
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def strength(self) -> float:
        """||mean gradient||^2."""
        if self._sum is None or self._count == 0:
            return 0.0
        inv = 1.0 / self._count
        return float(
            sum(np.square(a * inv).sum() for a in self._sum)
        )

    @property
    def reliability(self) -> float:
        """||mean g||^2 / mean ||g_i||^2, in [0, 1]; 0 if no signal."""
        if self._count == 0 or self._sum_sq_norms <= 0.0:
            return 0.0
        return self.strength / (self._sum_sq_norms / self._count)

    @property
    def readiness(self) -> float:
        """strength * reliability = ||mean g||^4 / mean ||g_i||^2."""
        return self.strength * self.reliability

    def as_dict(self) -> dict:
        return {
            "readiness": self.readiness,
            "readiness_strength": self.strength,
            "readiness_reliability": self.reliability,
            "readiness_micro_batches": self._count,
        }


def top_hessian_eigenvalue(
    hvp: Callable[[Array], Array],
    dim: int,
    n_iters: int = 20,
    seed: int = 0,
    tol: float = 1e-6,
) -> tuple[float, int]:
    """Dominant Hessian eigenvalue by power iteration over an HVP callable.

    Returns ``(eigenvalue, iterations_used)`` where the eigenvalue is the one
    of largest magnitude, signed via the Rayleigh quotient. The caller supplies
    ``hvp`` (e.g. JAX: ``lambda v: jvp(grad(loss), (p,), (unflatten(v),))[1]``
    flattened back), so this module stays framework-free. Rising sharpness is
    a supporting, optional diagnostic in PLAN.md.

    Args:
        hvp: maps a flat vector of length ``dim`` to (Hessian @ vector).
        dim: flattened parameter dimension.
        n_iters: maximum power-iteration steps (10-20 is typical).
        seed: seed for the random start vector.
        tol: relative tolerance on successive Rayleigh quotients.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    eig = 0.0
    for it in range(1, n_iters + 1):
        hv = np.asarray(hvp(v), dtype=np.float64).reshape(-1)
        if hv.shape[0] != dim:
            raise ValueError(f"hvp returned length {hv.shape[0]}, expected {dim}")
        new_eig = float(v @ hv)
        norm = np.linalg.norm(hv)
        if norm == 0.0:
            return 0.0, it
        v = hv / norm
        if abs(new_eig - eig) <= tol * max(1.0, abs(new_eig)):
            return new_eig, it
        eig = new_eig
    return eig, n_iters
