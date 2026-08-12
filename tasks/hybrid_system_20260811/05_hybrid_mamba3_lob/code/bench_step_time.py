"""GPU microbenchmark: baseline vs hybrid trunk at production shape.

Deliberately data-free. The question here is only whether a 13,000-token
sequence survives a global attention layer on one GH200, and what the swap
costs per optimiser step. Mounting 48 SquashFS shards to answer that would add
a second failure mode without adding evidence.

Usage (attached to an idle allocation):
  srun --jobid=<ID> --overlap --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
       --cpu-bind=none <python> bench_step_time.py --arch hybrid_mamba3
"""
import argparse
import os
import statistics
import sys
import time

REPO = "/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811"
sys.path.insert(0, os.path.join(REPO, "src"))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import optax                                                  # noqa: E402
from flax.core import unfreeze                                # noqa: E402

from s5.registry import build_backbone                        # noqa: E402
from lob.lob_seq_model import BatchPaddedLobPredModel         # noqa: E402

VOCAB = 2112
D_BOOK = 503            # reproduces the published 33,610,439 baseline exactly
TOKENS_PER_MSG = 26


def model_args(arch, d_model, n_layers, attn_layers, attn_heads, attn_d_ff, flash):
    return argparse.Namespace(
        architecture=arch, ssm_type=None, model_type=None,
        d_model=d_model, n_layers=n_layers,
        mamba3_d_state=128, mamba3_expand=2, mamba3_headdim=64,
        mamba3_chunk_size=64, mamba3_rope_fraction=0.5,
        mamba3_use_triton=False, mamba3_use_cuda=False, tp_size=1,
        p_dropout=0.0, remat=False,
        hybrid_attn_layers=attn_layers,
        hybrid_attn_heads=attn_heads,
        hybrid_attn_d_ff=attn_d_ff,
        hybrid_attn_flash=flash,
        hybrid_attn_positional_encoding=False,
        attention_max_cache_len=26000,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mamba3")
    ap.add_argument("--d-model", type=int, default=640)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--msgs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--attn-layers", default=None)
    ap.add_argument("--attn-heads", type=int, default=10)
    ap.add_argument("--attn-d-ff", type=int, default=0)
    ap.add_argument("--no-flash", action="store_true")
    a = ap.parse_args()

    dev = jax.local_devices()[0]
    print(f"device: {dev.device_kind} ({dev.platform})  jax {jax.__version__}")
    if dev.platform != "gpu":
        print("WARNING: not on a GPU; timings are meaningless")

    seq = a.msgs * TOKENS_PER_MSG
    args = model_args(a.arch, a.d_model, a.n_layers, a.attn_layers,
                      a.attn_heads, a.attn_d_ff, not a.no_flash)
    bb = build_backbone(args, print_shapes=True)
    model = BatchPaddedLobPredModel(
        ssm=bb.layer_factory, attn_ssm=bb.attn_layer_factory,
        attn_layers=bb.attn_layers, d_output=VOCAB, d_model=a.d_model,
        d_book=D_BOOK, n_message_layers=2, n_fused_layers=a.n_layers,
        n_book_pre_layers=1, n_book_post_layers=1,
        activation="half_glu1", dropout=0.0, training=True,
        mode="none", prenorm=True, batchnorm=False)

    key = jax.random.PRNGKey(0)
    x_m = jax.random.randint(key, (a.batch, seq), 0, VOCAB, dtype=jnp.int32)
    x_b = jnp.zeros((a.batch, seq, D_BOOK), dtype=jnp.float32)
    t_m = jnp.zeros((a.batch, seq), dtype=jnp.float32)
    t_b = jnp.zeros((a.batch, seq), dtype=jnp.float32)
    y = jax.random.randint(key, (a.batch, seq), 0, VOCAB, dtype=jnp.int32)

    print(f"shape: batch={a.batch} seq={seq} ({a.msgs} msg x {TOKENS_PER_MSG} tok) "
          f"d_model={a.d_model} n_layers={a.n_layers}")
    t0 = time.time()
    variables = model.init(key, x_m, x_b, t_m, t_b, method="__call_ar__")
    params = unfreeze(variables)["params"]
    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    print(f"params: {n_params:,}   (init {time.time() - t0:.1f}s)")

    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)

    def loss_fn(p):
        logits = model.apply({"params": p}, x_m, x_b, t_m, t_b,
                             method="__call_ar__")
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(p, o):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, o = tx.update(grads, o, p)
        return optax.apply_updates(p, updates), o, loss

    t0 = time.time()
    params, opt_state, loss = step(params, opt_state)
    jax.block_until_ready(loss)
    print(f"compile + first step: {time.time() - t0:.1f}s   loss={float(loss):.4f}")

    for _ in range(a.warmup):
        params, opt_state, loss = step(params, opt_state)
    jax.block_until_ready(loss)

    times = []
    for i in range(a.iters):
        t0 = time.time()
        params, opt_state, loss = step(params, opt_state)
        jax.block_until_ready(loss)
        times.append(time.time() - t0)
        print(f"  step {i}: {times[-1]:.3f}s  loss={float(loss):.4f}")

    med = statistics.median(times)
    stats = dev.memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0) / 1e9
    print()
    print(f"RESULT arch={a.arch} params={n_params} "
          f"median_s_per_step={med:.4f} "
          f"tokens_per_s={a.batch * seq / med:.0f} "
          f"peak_hbm_gb={peak:.1f} "
          f"attn_layers={bb.attn_layers}")


if __name__ == "__main__":
    main()
