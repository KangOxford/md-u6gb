"""Structural smoke test: build the baseline and the hybrid trunk side by side.

Runs on CPU at toy width. It answers three questions before any GPU time is
spent: does a heterogeneous stack initialise at all, does the attention layer
land where the Nemotron rule says it should, and what does it cost in
parameters relative to the recurrent layer it displaced.
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = "/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811"
sys.path.insert(0, os.path.join(REPO, "src"))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from flax.core import unfreeze                                # noqa: E402

from s5.registry import build_backbone, nemotron_attention_indices  # noqa: E402
from lob.lob_seq_model import BatchPaddedLobPredModel         # noqa: E402


def stub_args(architecture, d_model, n_layers, **over):
    ns = argparse.Namespace(
        architecture=architecture, ssm_type=None, model_type=None,
        d_model=d_model, n_layers=n_layers,
        mamba3_d_state=16, mamba3_expand=2, mamba3_headdim=8,
        mamba3_chunk_size=8, mamba3_rope_fraction=0.5,
        mamba3_use_triton=False, mamba3_use_cuda=False, tp_size=1,
        p_dropout=0.0, remat=False,
        hybrid_attn_heads=4, hybrid_attn_flash=False,
        hybrid_attn_positional_encoding=False,
        attention_max_cache_len=512,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def count(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def build(architecture, d_model, n_layers, d_output, d_book, n_msg, seq_msg, book_len):
    args = stub_args(architecture, d_model, n_layers)
    backbone = build_backbone(args)
    model = BatchPaddedLobPredModel(
        ssm=backbone.layer_factory,
        attn_ssm=backbone.attn_layer_factory,
        attn_layers=backbone.attn_layers,
        d_output=d_output,
        d_model=d_model,
        d_book=d_book,
        n_message_layers=n_msg,
        n_fused_layers=n_layers,
        n_book_pre_layers=1,
        n_book_post_layers=1,
        activation="half_glu1",
        dropout=0.0,
        training=False,
        mode="none",
        prenorm=True,
        batchnorm=False,
    )
    b = 1
    x_m = jnp.zeros((b, seq_msg), dtype=jnp.int32)
    # The dataloader already repeats each book row across its message tokens,
    # so the book arrives at message-token length, not message length.
    x_b = jnp.zeros((b, book_len, d_book), dtype=jnp.float32)
    t_m = jnp.zeros((b, seq_msg), dtype=jnp.float32)
    t_b = jnp.zeros((b, book_len), dtype=jnp.float32)
    # Training goes through __call_ar__, not the legacy pooled __call__.
    variables = model.init(jax.random.PRNGKey(0), x_m, x_b, t_m, t_b,
                           method="__call_ar__")
    params = unfreeze(variables)["params"]
    out = model.apply(variables, x_m, x_b, t_m, t_b, method="__call_ar__")
    return backbone, params, count(params), out.shape


def per_layer(params, stack="fused_s5"):
    """Parameter count of each layer in one stack, plus what kind it is."""
    rows = []
    layers = params[stack]
    i = 0
    while f"layers_{i}" in layers:
        blk = layers[f"layers_{i}"]
        # SequenceLayer always names its mixer "seq"; the layer kind is only
        # visible one level down, in which parameter groups that mixer created.
        inner = sorted(blk.get("seq", blk).keys())
        kind = "ATTENTION" if "attn" in inner else "mamba3"
        rows.append((i, kind, count(blk), inner))
        i += 1
    return rows


if __name__ == "__main__":
    D, L = 64, 6
    D_OUT, D_BOOK, N_MSG = 32, 20, 2
    MSGS, TOK = 8, 26
    SEQ = MSGS * TOK
    BOOK_LEN = SEQ

    print(f"toy config: d_model={D} n_fused_layers={L} vocab={D_OUT} "
          f"seq={SEQ} ({MSGS} msg x {TOK} tok)")
    print(f"nemotron rule at L={L}: attention at {nemotron_attention_indices(L)}")
    print()

    results = {}
    for arch in ("mamba3", "hybrid_mamba3"):
        bb, params, n, shape = build(arch, D, L, D_OUT, D_BOOK, N_MSG, SEQ, BOOK_LEN)
        results[arch] = (bb, params, n, shape)
        print(f"[{arch}] built ok  params={n:,}  out={shape}  "
              f"hybrid={bb.is_hybrid} attn_layers={bb.attn_layers}")

    print()
    print("fused trunk, layer by layer")
    print(f"{'layer':>5}  {'baseline':>18}  {'hybrid':>18}")
    print("-" * 48)
    base_rows = per_layer(results["mamba3"][1])
    hyb_rows = per_layer(results["hybrid_mamba3"][1])
    for (i, bk, bn, _), (_, hk, hn, _) in zip(base_rows, hyb_rows):
        mark = "  <<<" if hk == "ATTENTION" else ""
        print(f"{i:>5}  {bk:>9} {bn:>8,}  {hk:>9} {hn:>8,}{mark}")

    nb, nh = results["mamba3"][2], results["hybrid_mamba3"][2]
    print()
    print(f"total params  baseline {nb:,}   hybrid {nh:,}   "
          f"delta {nh - nb:+,} ({100.0 * (nh - nb) / nb:+.2f}%)")

    # The two stacks must be identical everywhere except the swapped positions.
    attn = set(results["hybrid_mamba3"][0].attn_layers)
    same = all(bn == hn for (i, _, bn, _), (_, _, hn, _) in zip(base_rows, hyb_rows)
               if i not in attn)
    print(f"non-attention layers identical in size: {same}")
    print()
    print("parameter groups of the swapped layer")
    for (i, k, _, inner) in hyb_rows:
        if k == "ATTENTION":
            print(f"  hybrid layer {i}: {inner}")
    for (i, k, _, inner) in base_rows:
        if i in attn:
            print(f"  baseline layer {i}: {inner}")

    # Parameter counts do not depend on sequence length here (the sinusoidal PE
    # is not learned, and hybrid switches it off anyway), so the production
    # width can be measured with a two-message sequence.
    print()
    print("=" * 62)
    print("production width: d_model=640, n_layers=6, vocab=2112, book=503")
    print("=" * 62)
    real = {}
    for arch in ("mamba3", "hybrid_mamba3"):
        args = stub_args(arch, 640, 6, mamba3_d_state=128, mamba3_expand=2,
                         mamba3_headdim=64, mamba3_chunk_size=64,
                         hybrid_attn_heads=10, hybrid_attn_flash=False,
                         attention_max_cache_len=26000)
        bb = build_backbone(args)
        model = BatchPaddedLobPredModel(
            ssm=bb.layer_factory, attn_ssm=bb.attn_layer_factory,
            attn_layers=bb.attn_layers, d_output=2112, d_model=640,
            d_book=503, n_message_layers=2, n_fused_layers=6,
            n_book_pre_layers=1, n_book_post_layers=1,
            activation="half_glu1", dropout=0.0, training=False,
            mode="none", prenorm=True, batchnorm=False)
        n_tok = 2 * 26
        v = model.init(jax.random.PRNGKey(0),
                       jnp.zeros((1, n_tok), jnp.int32),
                       jnp.zeros((1, n_tok, 503), jnp.float32),
                       jnp.zeros((1, n_tok), jnp.float32),
                       jnp.zeros((1, n_tok), jnp.float32),
                       method="__call_ar__")
        pr = unfreeze(v)["params"]
        real[arch] = (bb, pr, count(pr))
        print(f"[{arch}] params={count(pr):,}  attn_layers={bb.attn_layers}")

    rb, rh = real["mamba3"][2], real["hybrid_mamba3"][2]
    print(f"delta {rh - rb:+,} ({100.0 * (rh - rb) / rb:+.2f}%)")
    PUBLISHED = 33_610_439
    print(f"published baseline checkpoint: {PUBLISHED:,} "
          f"-> stub reproduces it exactly: {rb == PUBLISHED}")

    # A param-matched arm answers "was it the attention or just the extra
    # parameters", by shrinking the attention block FFN until the swapped layer
    # costs what the mamba3 layer it replaced cost.
    hyb_layers = per_layer(real["hybrid_mamba3"][1])
    base_layers = per_layer(real["mamba3"][1])
    a_idx = real["hybrid_mamba3"][0].attn_layers[0]
    attn_cost = dict((i, n) for i, _, n, _ in hyb_layers)[a_idx]
    mamba_cost = dict((i, n) for i, _, n, _ in base_layers)[a_idx]
    d_ff_used = 4 * 640
    fixed = attn_cost - (2 * 640 + 1) * d_ff_used - 640
    d_ff_match = round((mamba_cost - fixed - 640) / (2 * 640 + 1))
    print(f"attention layer {attn_cost:,} vs mamba3 layer {mamba_cost:,} "
          f"(d_ff={d_ff_used})")
    print(f"param-matched d_ff would be ~{d_ff_match} "
          f"(predicted layer cost {fixed + (2 * 640 + 1) * d_ff_match + 640:,})")
    print()
    print("fused trunk at production width")
    print(f"{'layer':>5}  {'baseline':>18}  {'hybrid':>18}")
    print("-" * 48)
    for (i, bk, bn, _), (_, hk, hn, _) in zip(per_layer(real["mamba3"][1]),
                                              per_layer(real["hybrid_mamba3"][1])):
        mark = "  <<<" if hk == "ATTENTION" else ""
        print(f"{i:>5}  {bk:>9} {bn:>8,}  {hk:>9} {hn:>8,}{mark}")
