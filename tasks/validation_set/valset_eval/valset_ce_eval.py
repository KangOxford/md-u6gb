#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""valset_v1 固定验证集 CE 评测：33 个 scaling-law terminal checkpoints ×
val_subset_30720（实体 squashfs 包，样本即文件、无随机偏移）。

与 leakage_exp/leakage_test.py 同源的 eval 路径（metadata 合并 → init →
restore → jit_eval），三点不同：
  1) 数据源是 valset 实体包（train_size 必须 == 30,720），rand_offset=False；
  2) 全量顺序 sampler（loader 顺序 = ticker → date → id，确定可复现）；
  3) 按 manifest 循环 33 个 checkpoint，逐个落 json（已存在则跳过 → 断点续跑）。

结构约束：spawn DataLoader worker 会 re-import 主模块 —— 一切执行流必须在
if __name__ == "__main__" 之内，模块层只允许 import 与函数定义（否则 12 个
worker 各自触发 JAX CUDA backend init，显存互踩）。

用法（exp_R1_Mamba3 目录、训练 env、CUDA_VISIBLE_DEVICES 已限定空闲卡）:
  python valset_ce_eval.py --manifest manifest_33ckpt.json \
      --data_root <valset-mount> --out_dir results_<ts> --num_devices 3
"""
import os, sys, json, time, argparse
import numpy as onp

EXP_DIR = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
CONSTITUENTS = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/snp500_constituents_20260131.csv"
N_EXPECT = 30_720
sys.path.insert(0, EXP_DIR)


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data_root", required=True, help="valset squashfs 挂载点")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_devices", type=int, default=3)
    p.add_argument("--n_data_workers", type=int, default=12)  # spawn（fork 与 JAX 死锁）
    p.add_argument("--only", default=None, help="逗号分隔 label 白名单（smoke 用）")
    p.add_argument("--provenance",
                   default="/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/provenance_valset_v1_30720.npz",
                   help="用于确定包内实际存在的 ticker 集合（.npz 取 msg_paths；.npy 视为纯 ticker 数组）")
    p.add_argument("--sampler_indices", default=None,
                   help="npy：dataset 全局窗口索引集合（默认 range(n_expect)，即样本即文件的 valset 模式）")
    p.add_argument("--n_expect", type=int, default=30720,
                   help="dataset train_size 断言值（valset 包=30720；Jan-2026 全月池=7507307）")
    p.add_argument("--date_range", default="2022-01-01,2025-12-31",
                   help="dataset 日期范围（Jan-2026 评测用 2026-01-02,2026-01-31）")
    return p.parse_args()


def read_tickers(path):
    out = []
    for i, line in enumerate(open(path)):
        c = line.split(",")[0].strip()
        if not c: continue
        if i == 0 and c.lower().startswith("ticker"): continue
        out.append(c)
    return out


def tickers_in_valset(prov_path, universe):
    """确定数据源实际存在的 ticker 集合（dataloader 对缺席 ticker 会 assert）。
    .npz：valset provenance，取 msg_paths 的目录名；.npy：纯 ticker 数组（Jan pool）。"""
    if prov_path.endswith(".npz"):
        prov = onp.load(prov_path)
        present = {str(pth).rsplit("/", 3)[-2] for pth in prov["msg_paths"]}
    else:
        present = set(str(t) for t in onp.load(prov_path))
    kept = [t for t in universe if t in present]
    print(f"[tickers] {len(kept)}/{len(universe)} present in data source "
          f"(absent: {sorted(set(universe) - present)})", flush=True)
    return kept


def build_base_args(cli):
    class A: pass
    args = A()
    args.__dict__.update(dict(
        dir_name="/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500",
        data_root=cli.data_root,
        tickers=tickers_in_valset(cli.provenance, read_tickers(CONSTITUENTS)),
        train_date_range=tuple(cli.date_range.split(",")), test_date_range=None,
        token_mode="26tok", msg_seq_len=500, use_book_data=True, use_simple_book=False,
        book_transform=True, book_depth=500, book_ablation="real",
        masking="none", merging="padded", dataset="lobster-prediction",
        n_data_workers=cli.n_data_workers, prefetch_factor=2,
        random_offsets_train=False, shuffle_train=False, val_split=0.0, debug_overfit=False,
        micro_bsz=8, num_devices=cli.num_devices,
        jax_seed=42, restore=None, restore_step=None,
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
    return args


def setup_checkpoint(jax, args, base_args, mesh, restore_path, restore_step, ds_meta):
    """metadata 合并 → init → restore（param-only 回退）。返回 (state, val_model, jit_eval)。"""
    from lob.init_train import load_metadata, init_train_state, load_checkpoint
    from lob.sharding_utils import create_state_shardings
    from lob.train_helpers import create_jit_eval_step
    n_classes, seq_len, book_dim, book_seq_len, train_size = ds_meta
    args.__dict__.clear(); args.__dict__.update(base_args)  # 重置公共基线
    args.restore_step = restore_step
    ckpt_args = load_metadata(restore_path)
    for k, v in vars(ckpt_args).items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    for k in ['ssm_type','d_model','n_layers','blocks','ssm_size_base',
              'n_message_layers','n_book_pre_layers','n_book_post_layers',
              'mamba3_d_state','mamba3_expand','mamba3_headdim','mamba3_chunk_size',
              'mamba3_rope_fraction','mamba3_use_triton','activation_fn']:
        if hasattr(ckpt_args, k):
            setattr(args, k, getattr(ckpt_args, k))
    args.opt_config = "standard"
    print(f"[Metadata] ssm_type={args.ssm_type} d_model={args.d_model} "
          f"n_layers={args.n_layers} step={restore_step}", flush=True)
    state, model_cls = init_train_state(
        args, n_classes=n_classes, seq_len=seq_len, book_dim=book_dim,
        book_seq_len=book_seq_len, train_size=train_size, print_shapes=False)
    try:
        ckpt = load_checkpoint(state, restore_path, step=restore_step,
                               mesh=mesh, train=False, partial_restore=True)
        state = ckpt['model']
    except (ValueError, TypeError) as e:
        print(f"[Restore] StandardRestore failed ({e}); param-only fallback", flush=True)
        import orbax.checkpoint as ocp
        m = ocp.CheckpointManager(os.path.abspath(restore_path),
                                  item_names=('state', 'metadata'),
                                  options=ocp.CheckpointManagerOptions())
        step = restore_step or m.latest_step()
        r = m.restore(step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state.params)))
        state = state.replace(params=r['state'])
    state = jax.device_get(state)
    state = jax.device_put(state, create_state_shardings(state, mesh))
    val_model = model_cls(training=False, step_rescale=1)
    jit_eval = create_jit_eval_step(mesh, state, has_book_data=True)
    return state, val_model, jit_eval


def eval_full(jax, cli, mesh, ds_obj, ds_meta, state, val_model, jit_eval, label, eval_bsz,
              sampler_ids):
    """固定样本集合顺序评测（30,720 个）；返回 (per-sample loss, per-sample acc)。"""
    from lob.lobster_dataloader import LOBSTER
    from lob.dataloading import force_cpu
    from lob.sharding_utils import get_data_shardings_for_batch
    from lob.train_helpers import prep_batch
    n_classes, seq_len, book_dim, book_seq_len, train_size = ds_meta
    bsz = eval_bsz * cli.num_devices
    assert len(sampler_ids) % bsz == 0, f"bsz {bsz} 不整除 {len(sampler_ids)}（会丢样本）"
    from torch.utils.data import DataLoader as TorchDataLoader
    kw = dict(batch_size=bsz, sampler=sampler_ids, shuffle=False,
              drop_last=True, collate_fn=LOBSTER._collate_fn, pin_memory=True)
    if cli.n_data_workers > 0:
        kw.update(num_workers=cli.n_data_workers, multiprocessing_context="spawn",
                  prefetch_factor=6, persistent_workers=False, worker_init_fn=force_cpu)
    try:
        loader = TorchDataLoader(ds_obj.dataset_train, **kw)
    except Exception as e:
        print(f"[loader] spawn workers failed ({e}); num_workers=0", flush=True)
        kw = {k: v for k, v in kw.items() if k in ("batch_size", "sampler", "shuffle",
                                                    "drop_last", "collate_fn", "pin_memory")}
        loader = TorchDataLoader(ds_obj.dataset_train, **kw)
    # spawn worker unpickle dataset 时内嵌的 jax.Array 会触发 device_put；
    # 迭代器创建（= worker 启动）期间强制 JAX_PLATFORMS=cpu，worker 全程留在 CPU
    # （worker_init_fn=force_cpu 在 unpickle 之后才跑，来不及）。父进程 backend 已缓存，不受影响。
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        loader_it = iter(loader)
    finally:
        os.environ.pop("JAX_PLATFORMS", None)
    n_batches = len(sampler_ids) // bsz
    losses, accs = [], []
    t0 = time.time()
    for bi in range(n_batches):
        batch = next(loader_it)
        inputs, labels, its = prep_batch(batch, seq_len, cli.num_devices)
        in_sh, lb_sh, ts_sh = get_data_shardings_for_batch(mesh, has_book_data=(len(inputs) > 1))
        inputs = tuple(jax.make_array_from_process_local_data(sh, x) for x, sh in zip(inputs, in_sh))
        labels = jax.make_array_from_process_local_data(lb_sh, labels)
        its = tuple(jax.make_array_from_process_local_data(sh, t) for t, sh in zip(its, ts_sh))
        loss, acc, _ = jit_eval(inputs, labels, its, state, val_model.apply,
                                False, '__call_ar__', (onp.array([0])), False)
        # eval_step 返回 (B, flat_positions) 的 per-token CE/acc —— 对 token 维取均值
        # 得 per-sample 值；等长序列下 mean-of-sample-means == token 级全局 mean。
        losses.append(onp.asarray(jax.device_get(loss)).mean(axis=-1))
        accs.append(onp.asarray(jax.device_get(acc)).mean(axis=-1))
        if bi % 100 == 0:
            print(f"  [{label}] batch {bi}/{n_batches} loss={losses[-1].mean():.5f} "
                  f"({(time.time()-t0)/(bi+1):.2f}s/b)", flush=True)
    return onp.concatenate(losses), onp.concatenate(accs)


def main():
    cli = parse_cli()
    base = build_base_args(cli)

    import jax
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join(os.environ.get("TMPDIR", "/tmp"), "jax_valset_cache"))
    print(f"[jax] devices: {jax.device_count()}", flush=True)
    assert jax.device_count() == cli.num_devices, \
        f"visible devices {jax.device_count()} != --num_devices {cli.num_devices}"

    from lob.lobster_dataloader import LOBSTER_Dataset
    from lob.dataloading import create_lobster_prediction_dataset
    from lob.sharding_utils import initialize_mesh

    (ds_obj, _tr, _va, _te, _aux, n_classes, seq_len, in_dim, book_seq_len, book_dim,
     train_size) = create_lobster_prediction_dataset(
        base.dir_name, seed=42, mask_fn=LOBSTER_Dataset.no_mask,
        msg_seq_len=500, micro_bsz=base.micro_bsz, num_devices=base.num_devices,
        use_book_data=True, use_simple_book=False, book_transform=True,
        book_depth=500, book_ablation="real", n_data_workers=0,
        shuffle_train=False, rand_offset=False, debug_overfit=False,
        val_split=0.0, test_split=0.0, pin_memory=False, prefetch_factor=None,
        persistent_workers=False, test_dir_name=None, use_distributed_sampler=False,
        process_rank=0, process_count=1, tickers=base.tickers,
        data_root=base.data_root, train_date_range=list(base.train_date_range),
        test_date_range=None, token_mode="26tok")
    assert train_size == cli.n_expect, \
        f"train_size={train_size} != n_expect={cli.n_expect}（窗口枚举与清单假设不符）"
    print(f"[dataset] N={train_size:,} seq_len={seq_len} book_dim={book_dim}", flush=True)
    ds_meta = (n_classes, seq_len, book_dim, book_seq_len, train_size)
    if cli.sampler_indices:
        sampler_ids = [int(i) for i in onp.load(cli.sampler_indices)]
        assert max(sampler_ids) < train_size
        print(f"[sampler] {len(sampler_ids):,} indices from {cli.sampler_indices}", flush=True)
    else:
        sampler_ids = list(range(N_EXPECT))

    mesh = initialize_mesh(cli.num_devices)
    base_args = dict(base.__dict__)

    rng = onp.random.default_rng(20260729)
    def boot_mean(a, B=20000, chunk=2000):
        # 30720 样本 × 20000 draws 整块索引矩阵 ~2.4GB —— 分块累积
        means = []
        for s in range(0, B, chunk):
            idx = rng.integers(0, len(a), (min(chunk, B - s), len(a)))
            means.append(a[idx].mean(axis=1))
        m = onp.concatenate(means)
        return float(a.mean()), float(onp.percentile(m, 2.5)), float(onp.percentile(m, 97.5))

    manifest = json.load(open(cli.manifest))
    if cli.only:
        allow = set(cli.only.split(","))
        manifest = [m for m in manifest if m["label"] in allow]
    # LPT：大模型先抢，避免收尾阶段单卡拖一个 350M
    manifest.sort(key=lambda m: -int(m["num_params"]))
    os.makedirs(cli.out_dir, exist_ok=True)
    print(f"[plan] {len(manifest)} checkpoints -> {cli.out_dir}", flush=True)

    for m in manifest:
        label = m["label"]
        oj = os.path.join(cli.out_dir, f"valce_{label}.json")
        if os.path.exists(oj):
            print(f"[skip] {label}: done", flush=True)
            continue
        # 多 GPU 工作队列：mkdir 原子抢占（launcher 启动前清残锁）
        lock = os.path.join(cli.out_dir, f"lock_{label}")
        try:
            os.mkdir(lock)
        except FileExistsError:
            print(f"[skip] {label}: claimed by another worker", flush=True)
            continue
        eval_bsz = max(int(m["micro_bsz"]), 2)  # 200M/350M 训练 bsz=1 → 2（r4/r5 实测安全）
        print(f"\n===== {label} (jid {m['jid']}, step {m['step']}, eval_bsz {eval_bsz}/GPU) =====",
              flush=True)
        t0 = time.time()
        state, val_model, jit_eval = setup_checkpoint(
            jax, base, base_args, mesh, m["ckpt_dir"], int(m["step"]), ds_meta)
        losses, accs = eval_full(jax, cli, mesh, ds_obj, ds_meta,
                                 state, val_model, jit_eval, label, eval_bsz, sampler_ids)
        mean, lo, hi = boot_mean(losses)
        out = dict(label=label, size=m["size"], seed=m["seed"], jid=m["jid"],
                   ckpt_dir=m["ckpt_dir"], step=int(m["step"]),
                   num_params=m["num_params"], D_tokens=m["D_tokens"],
                   eval_bsz_per_gpu=eval_bsz, num_devices=cli.num_devices,
                   n_samples=int(len(losses)),
                   val_ce_mean=mean, val_ce_ci95=[lo, hi],
                   val_acc_mean=float(accs.mean()),
                   jan2026_ce=m["jan2026_ce"],
                   wall_sec=round(time.time() - t0, 1),
                   data=(cli.sampler_indices or "shard_valset_v1_30720.squashfs"),
                   protocol="sequential full pass, per-sample CE, drop_last exact")
        json.dump(out, open(oj, "w"), indent=1)
        onp.save(oj.replace(".json", "_sampleloss.npy"), losses.astype(onp.float32))
        print(json.dumps({k: out[k] for k in
                          ("label", "val_ce_mean", "val_ce_ci95", "val_acc_mean",
                           "jan2026_ce", "wall_sec")}, indent=1), flush=True)
        del state, jit_eval

    print("VALSET_CE_EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
