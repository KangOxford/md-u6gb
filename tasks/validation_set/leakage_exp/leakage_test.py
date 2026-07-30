#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""泄漏行为学实验：在 SEEN / MID / VAL 三组（各 30,720 样本）上测同一 checkpoint 的
per-token CE，bootstrap 置信区间比较。

判据（预注册，见 VALSET_V1_REPORT.md §10）：
  H1（检测力）：CE(SEEN) < CE(MID)，差值 95% CI 不含 0；
  H2（无泄漏）：CE(VAL) − CE(MID) 的 95% CI 包含 0。

用法:
  python leakage_test.py --restore checkpoints/j4499538_5vu8avcx_4499538 \
      --label 78M-s5 --micro_bsz 8 --data_root <comma-mounts> --out_json results/78M.json
"""
import os
import sys
import json
import time
import argparse
import numpy as onp

EXP_DIR = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
CONSTITUENTS = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/snp500_constituents_20260131.csv"
GROUPS_DIR = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/leakage_exp/groups"
N_EXPECT = 323_221_385

if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="LOB Scaling Law Leakage Evaluation")
    p.add_argument("--restore", required=True, help="逗号分隔的 checkpoint 目录列表")
    p.add_argument("--label", required=True, help="逗号分隔，与 --restore 对齐")
    p.add_argument("--micro_bsz", default="8", help="逗号分隔，与 --restore 对齐")
    p.add_argument("--num_devices", type=int, default=4)
    p.add_argument("--n_data_workers", type=int, default=0, help="Data loader workers (default: 0 to prevent multiprocessing lock/re-exec)")
    p.add_argument("--data_root", required=True, help="逗号分隔的 squashfs 挂载点目录")
    p.add_argument("--out_json", required=True, help="输出结果 JSON 路径")
    p.add_argument("--restore_step", type=int, default=None, help="指定恢复 step (默认 latest)")
    return p.parse_args()


def read_tickers(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            c = line.split(",")[0].strip()
            if not c:
                continue
            if i == 0 and c.lower().startswith("ticker"):
                continue
            out.append(c)
    return out


class Struct:
    pass


def boot_mean(a, B=20000, rng_seed=20260730):
    rng = onp.random.default_rng(rng_seed)
    m = rng.choice(a, (B, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(onp.percentile(m, 2.5)), float(onp.percentile(m, 97.5))


def boot_diff(a, b, B=20000, rng_seed=20260730):
    rng = onp.random.default_rng(rng_seed)
    d = (rng.choice(a, (B, len(a)), replace=True).mean(axis=1)
         - rng.choice(b, (B, len(b)), replace=True).mean(axis=1))
    return float(a.mean() - b.mean()), float(onp.percentile(d, 2.5)), float(onp.percentile(d, 97.5))


def setup_checkpoint(args, mesh, n_classes, seq_len, book_dim, book_seq_len, train_size, restore_path):
    """Metadata merging -> init -> restore (with param-only fallback). Returns (state, val_model, jit_eval)."""
    from lob.init_train import load_metadata, init_train_state, load_checkpoint
    from lob.sharding_utils import create_state_shardings
    from lob.train_helpers import create_jit_eval_step
    import jax

    ckpt_args = load_metadata(restore_path)
    for k, v in vars(ckpt_args).items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    for k in ['ssm_type', 'd_model', 'n_layers', 'blocks', 'ssm_size_base',
              'n_message_layers', 'n_book_pre_layers', 'n_book_post_layers',
              'mamba3_d_state', 'mamba3_expand', 'mamba3_headdim', 'mamba3_chunk_size',
              'mamba3_rope_fraction', 'mamba3_use_triton', 'activation_fn']:
        if hasattr(ckpt_args, k):
            setattr(args, k, getattr(ckpt_args, k))
    args.opt_config = "standard"
    print(f"[Metadata] ssm_type={args.ssm_type} d_model={args.d_model} L={args.n_layers}", flush=True)

    state, model_cls = init_train_state(
        args, n_classes=n_classes, seq_len=seq_len, book_dim=book_dim,
        book_seq_len=book_seq_len, train_size=train_size, print_shapes=False
    )
    try:
        ckpt = load_checkpoint(state, restore_path, step=args.restore_step,
                               mesh=mesh, train=False, partial_restore=True)
        state = ckpt['model']
    except (ValueError, TypeError) as e:
        print(f"[Restore] StandardRestore failed ({e}); param-only fallback", flush=True)
        import orbax.checkpoint as ocp
        m = ocp.CheckpointManager(os.path.abspath(restore_path),
                                  item_names=('state', 'metadata'),
                                  options=ocp.CheckpointManagerOptions())
        step = args.restore_step or m.latest_step()
        r = m.restore(step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state.params)))
        state = state.replace(params=r['state'])

    state = jax.device_get(state)
    state = jax.device_put(state, create_state_shardings(state, mesh))
    val_model = model_cls(training=False, step_rescale=1)
    jit_eval = create_jit_eval_step(mesh, state, has_book_data=True)
    return state, val_model, jit_eval


def eval_group(args, ds_obj, mesh, state, val_model, jit_eval, seq_len, name, idx_path):
    from torch.utils.data import DataLoader as TorchDataLoader
    from lob.lobster_dataloader import LOBSTER
    from lob.dataloading import force_cpu
    from lob.sharding_utils import get_data_shardings_for_batch
    from lob.train_helpers import prep_batch
    import jax

    idx = onp.load(idx_path)
    bsz = args.micro_bsz * args.num_devices
    kw = dict(batch_size=bsz, sampler=[int(i) for i in idx], shuffle=False,
              drop_last=True, collate_fn=LOBSTER._collate_fn, pin_memory=True)
    if args.n_data_workers > 0:
        kw.update(num_workers=args.n_data_workers, multiprocessing_context="spawn",
                  prefetch_factor=6, persistent_workers=False, worker_init_fn=force_cpu)

    try:
        loader = TorchDataLoader(ds_obj.dataset_train, **kw)
    except Exception as e:
        print(f"[loader] spawn workers failed ({e}); falling back to num_workers=0", flush=True)
        kw = {k: v for k, v in kw.items() if k in ("batch_size", "sampler", "shuffle",
                                                    "drop_last", "collate_fn", "pin_memory")}
        loader = TorchDataLoader(ds_obj.dataset_train, **kw)

    losses = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        inputs, labels, its = prep_batch(batch, seq_len, args.num_devices)
        in_sh, lb_sh, ts_sh = get_data_shardings_for_batch(mesh, has_book_data=(len(inputs) > 1))
        inputs = tuple(jax.make_array_from_process_local_data(sh, x) for x, sh in zip(inputs, in_sh))
        labels = jax.make_array_from_process_local_data(lb_sh, labels)
        its = tuple(jax.make_array_from_process_local_data(sh, t) for t, sh in zip(its, ts_sh))
        loss, acc, _ = jit_eval(inputs, labels, its, state, val_model.apply,
                                False, '__call_ar__', (onp.array([0])), False)
        # Safely convert loss tensor array to Python scalar mean
        batch_loss_val = float(onp.asarray(jax.device_get(loss)).mean())
        losses.append(batch_loss_val)
        if bi % 100 == 0:
            print(f"  [{name}] batch {bi}/{len(loader)} loss={losses[-1]:.5f} "
                  f"({(time.time()-t0)/(bi+1):.2f}s/b)", flush=True)

    arr = onp.array(losses)
    print(f"[{name}] batches={len(arr)} meanCE={arr.mean():.6f}", flush=True)
    return arr


def main():
    args_cli = parse_args()

    args = Struct()
    args.__dict__.update(dict(
        dir_name="/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500",
        data_root=args_cli.data_root, tickers=read_tickers(CONSTITUENTS),
        train_date_range=("2022-01-01", "2025-12-31"), test_date_range=None,
        token_mode="26tok", msg_seq_len=500, use_book_data=True, use_simple_book=False,
        book_transform=True, book_depth=500, book_ablation="real",
        masking="none", merging="padded", dataset="lobster-prediction",
        n_data_workers=args_cli.n_data_workers, prefetch_factor=2,
        random_offsets_train=True, shuffle_train=True, val_split=0.0, debug_overfit=False,
        micro_bsz=8, num_devices=args_cli.num_devices,
        jax_seed=42, restore=None, restore_step=args_cli.restore_step,
        d_model=1024, n_layers=6, blocks=16, ssm_size_base=1024, ssm_type="mamba3",
        n_message_layers=2, n_book_pre_layers=1, n_book_post_layers=1,
        activation_fn="half_glu1", C_init="trunc_standard_normal", conj_sym=True,
        clip_eigs=True, bidirectional=False, prenorm=True, batchnorm=False,
        bn_momentum=0.95, dt_min=0.001, dt_max=0.1, dt_global=False,
        discretization="zoh", mode="none", p_dropout=0.0,
        mamba3_d_state=128, mamba3_expand=2, mamba3_headdim=64, mamba3_chunk_size=64,
        mamba3_rope_fraction=0.5, mamba3_use_triton=False,
        ssm_lr_base=5e-4, lr_factor=1, warmup_end=0.01, weight_decay=0.005,
        max_grad_norm=1.0, opt_config="standard", cosine_anneal=True, lr_min=0,
        lr_patience=4, reduce_factor=0.9, epochs=1, early_stop_patience=1000,
        muon_lr=0.02, muon_wd=None, mini_epochs=1, grad_accum_steps=1, local_steps_k=0,
        curtail_epochs=None, hierarchical=False, ignore_times=False,
        process_index=0, process_count=1, is_distributed=False, remat=False,
        dtype="bfloat16", use_flash=True, use_rope=True, rope_base=10000, n_heads=4, d_ff=1024,
    ))

    import jax
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join(os.environ.get("TMPDIR", "/tmp"), "jax_leak_cache"))
    print(f"[jax] devices: {jax.device_count()}", flush=True)

    from lob.lobster_dataloader import LOBSTER_Dataset
    from lob.dataloading import create_lobster_prediction_dataset
    from lob.sharding_utils import initialize_mesh

    (ds_obj, _tr, _va, _te, _aux, n_classes, seq_len, in_dim, book_seq_len, book_dim,
     train_size) = create_lobster_prediction_dataset(
        args.dir_name, seed=42, mask_fn=LOBSTER_Dataset.no_mask,
        msg_seq_len=500, micro_bsz=args.micro_bsz, num_devices=args.num_devices,
        use_book_data=True, use_simple_book=False, book_transform=True,
        book_depth=500, book_ablation="real", n_data_workers=0,
        shuffle_train=True, rand_offset=True, debug_overfit=False,
        val_split=0.0, test_split=0.0, pin_memory=False, prefetch_factor=None,
        persistent_workers=False, test_dir_name=None, use_distributed_sampler=False,
        process_rank=0, process_count=1, tickers=args.tickers,
        data_root=args.data_root, train_date_range=list(args.train_date_range),
        test_date_range=None, token_mode="26tok"
    )
    assert train_size == N_EXPECT, f"train_size={train_size} != {N_EXPECT}"
    print(f"[dataset] N={train_size:,} seq_len={seq_len} book_dim={book_dim}", flush=True)

    mesh = initialize_mesh(args.num_devices)

    restores = args_cli.restore.split(",")
    labels = args_cli.label.split(",")
    bszs = [int(b) for b in str(args_cli.micro_bsz).split(",")]
    assert len(restores) == len(labels) == len(bszs), "Mismatch in restore, label, micro_bsz count"

    for restore_path, label, bsz_i in zip(restores, labels, bszs):
        args.micro_bsz = bsz_i
        print(f"\n===== checkpoint {label} (bsz {bsz_i}/GPU) =====", flush=True)
        state, val_model, jit_eval = setup_checkpoint(args, mesh, n_classes, seq_len, book_dim, book_seq_len, train_size, restore_path)
        results = {}
        for name in ["mid", "seen", "val"]:
            results[name] = eval_group(
                args, ds_obj, mesh, state, val_model, jit_eval, seq_len, name,
                os.path.join(GROUPS_DIR, f"group_{name}.npy")
            )

        mean_s, lo_s, hi_s = boot_mean(results["seen"])
        mean_m, lo_m, hi_m = boot_mean(results["mid"])
        mean_v, lo_v, hi_v = boot_mean(results["val"])
        d_sm_m, d_sm_lo, d_sm_hi = boot_diff(results["seen"], results["mid"])
        d_vm_m, d_vm_lo, d_vm_hi = boot_diff(results["val"], results["mid"])

        out = dict(
            label=label, restore=restore_path, step=int(state.step),
            micro_bsz=bsz_i, group_size=30720,
            n_batches={k: int(len(v)) for k, v in results.items()},
            ce=dict(
                seen=dict(mean=mean_s, lo=lo_s, hi=hi_s),
                mid=dict(mean=mean_m, lo=lo_m, hi=hi_m),
                val=dict(mean=mean_v, lo=lo_v, hi=hi_v)
            ),
            diff_seen_minus_mid=dict(mean=d_sm_m, lo=d_sm_lo, hi=d_sm_hi),
            diff_val_minus_mid=dict(mean=d_vm_m, lo=d_vm_lo, hi=d_vm_hi),
            H1_memorization_gap_detected=bool(d_sm_hi < 0),
            H2_val_indistinguishable_from_unseen=bool(d_vm_lo <= 0 <= d_vm_hi)
        )

        oj = args_cli.out_json.replace(".json", f"_{label}.json")
        os.makedirs(os.path.dirname(oj), exist_ok=True)
        with open(oj, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)

        for k in ("seen", "mid", "val"):
            onp.save(oj.replace(".json", f"_{k}_batchloss.npy"), results[k])
        print(json.dumps(out, indent=1), flush=True)
        del state, jit_eval

    print("LEAKAGE_TEST_OK", flush=True)


if __name__ == "__main__":
    main()
