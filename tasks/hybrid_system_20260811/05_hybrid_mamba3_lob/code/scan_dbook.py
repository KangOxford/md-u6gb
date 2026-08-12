"""Find the d_book that reproduces the published baseline parameter count."""
import os, sys, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = "/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811"
sys.path.insert(0, os.path.join(REPO, "src"))
import jax, jax.numpy as jnp
from flax.core import unfreeze
from s5.registry import build_backbone
from lob.lob_seq_model import BatchPaddedLobPredModel

TARGET = 33_610_439

def n_params(d_book, n_msg=2, d_model=640, n_layers=6, vocab=2112):
    args = argparse.Namespace(
        architecture="mamba3", ssm_type=None, model_type=None,
        d_model=d_model, n_layers=n_layers,
        mamba3_d_state=128, mamba3_expand=2, mamba3_headdim=64,
        mamba3_chunk_size=64, mamba3_rope_fraction=0.5,
        mamba3_use_triton=False, mamba3_use_cuda=False, tp_size=1,
        p_dropout=0.0, remat=False)
    bb = build_backbone(args)
    m = BatchPaddedLobPredModel(
        ssm=bb.layer_factory, attn_ssm=None, attn_layers=(),
        d_output=vocab, d_model=d_model, d_book=d_book,
        n_message_layers=n_msg, n_fused_layers=n_layers,
        n_book_pre_layers=1, n_book_post_layers=1,
        activation="half_glu1", dropout=0.0, training=False,
        mode="none", prenorm=True, batchnorm=False)
    t = 26
    v = m.init(jax.random.PRNGKey(0),
               jnp.zeros((1, t), jnp.int32), jnp.zeros((1, t, d_book)),
               jnp.zeros((1, t)), jnp.zeros((1, t)), method="__call_ar__")
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(unfreeze(v)["params"]))

if __name__ == "__main__":
    print(f"target (published baseline) = {TARGET:,}")
    for db in (500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512):
        n = n_params(db)
        hit = "  <<< MATCH" if n == TARGET else ""
        print(f"  d_book={db:>4}  params={n:>12,}  delta={n - TARGET:>+9,}{hit}")
