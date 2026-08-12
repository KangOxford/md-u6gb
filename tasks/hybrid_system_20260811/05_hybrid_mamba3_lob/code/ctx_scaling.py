"""How far does the context window stretch before it stops fitting in one GH200?

The reason to ask: 35.7% of all references in generated output point at an order
that is older than the 250-message conditioning window, and recall in that bin
is 39.2% against 74.9% overall. Those references are unreachable by construction,
not badly modelled. Lengthening the window is the only thing that can reach them.

Estimating this is not good enough. The dominant activation term is the chunked
SSM state, which materialises B x (L/chunk) x heads x headdim x d_state per
layer, and at 4,000 messages a back-of-envelope lands between 40 and 80 GB
against 85.5 GB of HBM. That is inside the error bars of the assumptions, so the
answer has to be measured.

Measures peak device memory at production width for both the forward pass alone
and forward plus backward, since "fits" means different things for generation
and for training. Sweeps message counts and reports where it stops fitting.
"""
import argparse
import os
import sys
import time

REPO = "/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811"
sys.path.insert(0, os.path.join(REPO, "src"))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from s5.registry import build_backbone                        # noqa: E402
from lob.lob_seq_model import BatchPaddedLobPredModel         # noqa: E402

TOK_PER_MSG = 26
GH200_HBM_GB = 85.5


def prod_args(architecture, **over):
    """The configuration actually trained, not the toy one used by the smoke."""
    ns = argparse.Namespace(
        architecture=architecture, ssm_type=None, model_type=None,
        d_model=640, n_layers=6,
        mamba3_d_state=128, mamba3_expand=2, mamba3_headdim=64,
        mamba3_chunk_size=64, mamba3_rope_fraction=0.5,
        mamba3_use_triton=False, mamba3_use_cuda=False, tp_size=1,
        p_dropout=0.0, remat=False,
        hybrid_attn_heads=10, hybrid_attn_flash=True,
        hybrid_attn_positional_encoding=False,
        attention_max_cache_len=512,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def build_model(args, d_output=2112, d_book=500):
    backbone = build_backbone(args)
    model = BatchPaddedLobPredModel(
        ssm=backbone.layer_factory,
        attn_ssm=backbone.attn_layer_factory,
        attn_layers=backbone.attn_layers,
        d_output=d_output, d_model=args.d_model, d_book=d_book,
        n_message_layers=2, n_fused_layers=args.n_layers,
        n_book_pre_layers=1, n_book_post_layers=1,
        activation="half_glu1", dropout=0.0, training=True,
        mode="none", prenorm=True, batchnorm=False,
    )
    return backbone, model


def peak_gb():
    """Peak bytes the device allocator has handed out since process start.

    peak_bytes_in_use is monotonic and the PJRT client exposes no working reset,
    so a second config measured in the same process inherits the first one's
    high-water mark. Measuring 500 fwd, then 500 fwd+bwd, then 1000 fwd in one
    process reported 3.25, 18.02, 18.17, where the third number is the second
    one showing through. Each configuration therefore gets its own process, and
    main() refuses to measure more than one.
    """
    d = jax.devices()[0]
    s = d.memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 1e9


def measure(architecture, n_msgs, bsz, backward, d_book=500, remat=False):
    """Returns (peak_gb, seconds_per_call) or ('OOM', None)."""
    args = prod_args(architecture, remat=remat)
    backbone, model = build_model(args, d_book=d_book)
    seq = n_msgs * TOK_PER_MSG

    x_m = jnp.zeros((bsz, seq), dtype=jnp.int32)
    x_b = jnp.zeros((bsz, seq, d_book), dtype=jnp.float32)
    t_m = jnp.zeros((bsz, seq), dtype=jnp.float32)
    t_b = jnp.zeros((bsz, seq), dtype=jnp.float32)

    try:
        variables = model.init(jax.random.PRNGKey(0), x_m, x_b, t_m, t_b,
                               method="__call_ar__")
    except Exception as e:                       # init itself can exhaust HBM
        return f"OOM@init ({type(e).__name__})", None

    def loss_fn(p):
        out = model.apply({"params": p}, x_m, x_b, t_m, t_b, method="__call_ar__")
        return jnp.mean(out ** 2)

    fn = jax.jit(jax.grad(loss_fn)) if backward else jax.jit(loss_fn)
    params = variables["params"]

    try:
        r = fn(params)
        jax.block_until_ready(r)
    except Exception as e:
        msg = str(e)
        kind = "OOM" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) \
            else type(e).__name__
        return kind, None
    pk = peak_gb()

    t0 = time.perf_counter()
    for _ in range(3):
        r = fn(params)
    jax.block_until_ready(r)
    dt = (time.perf_counter() - t0) / 3
    return pk, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mamba3")
    ap.add_argument("--msgs", type=int, required=True,
                    help="one value only; peaks leak across configs in a process")
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--remat", action="store_true")
    ap.add_argument("--forward-only", action="store_true")
    a = ap.parse_args()

    bwd = not a.forward_only
    n = a.msgs
    pk, dt = measure(a.arch, n, a.bsz, bwd, remat=a.remat)
    label = "fwd+bwd" if bwd else "fwd"
    tag = f"{a.arch} msgs={n} tok={n*TOK_PER_MSG} bsz={a.bsz} remat={int(a.remat)} {label}"
    if isinstance(pk, str):
        print(f"ROW|{tag}|{pk}|—|—")
    else:
        print(f"ROW|{tag}|{pk:.2f}|{100*pk/GH200_HBM_GB:.1f}|{dt:.3f}")


if __name__ == "__main__":
    main()
