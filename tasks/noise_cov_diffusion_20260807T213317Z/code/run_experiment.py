#!/usr/bin/env python3
"""主实验: iid 噪声 vs 高维相关高斯噪声(HDGN) 的受控对照 + LOB-Bench 打分。

产出
----
results.json  逐 arm / 逐评估点 / 逐 NFE 的 LOB-Bench 分数与耗时
两条 quality-vs-cost 曲线, 用来判定「iid 最终能否追上 HDGN, 代价几何」
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, standardize, noise_covariance
from diffusion import NoiseSource, train_arm, ddim_sample, cosine_alphas
from lobbench_eval import score_all, COVERED, NOT_COVERED

DS = "/lus/lfs1aip2/projects/public/u6gb/datasets/lob_flat43_example_20260807T135612Z"
T8 = ["GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-gen", type=int, default=4000)
    ap.add_argument("--eval-nfe", type=int, default=50)
    ap.add_argument("--nfe-sweep", default="5,10,20,50,100")
    ap.add_argument("--eval-at", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    log(f"[setup] device={args.device}  T={args.T}  steps={args.steps}")
    X = load_dataset(DS, T8, T=args.T, stride=args.T, max_rows_per_ticker=args.rows, seed=args.seed)
    Xz, mu, sd = standardize(X)
    D = args.T * 43
    Xf = Xz.reshape(len(Xz), D).astype(np.float32)
    n_train = int(0.9 * len(Xf))
    Xtr, Xte = Xf[:n_train], Xf[n_train:]
    Sigma = noise_covariance(Xtr)
    w = np.linalg.eigvalsh(Sigma)[::-1]
    cond = float(w[0] / max(w[-1], 1e-12))
    log(f"[setup] 窗口 {X.shape} -> D={D}, train {len(Xtr)} / test {len(Xte)}, Σ 条件数 {cond:.3e}")

    real_F = (Xte.reshape(-1, args.T, 43) * sd + mu)          # 参照分布用 held-out
    eval_at = ([int(s) for s in args.eval_at.split(",") if s]
               or sorted({int(round(args.steps * f)) for f in
                          (0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.5, 0.72, 1.0)}))
    log(f"[setup] 评估点 {eval_at}")

    results = {"config": vars(args), "D": D, "sigma_cond": cond,
               "n_train": len(Xtr), "n_test": len(Xte),
               "features_covered": list(COVERED), "features_not_covered": list(NOT_COVERED),
               "arms": {}}

    # ---- 两个必需的对照基线, 否则分数无法解释 ----
    # (a) FLOOR: 真实 vs 真实(held-out 对半劈)。这是有限样本下分数能到的最好水平,
    #     任何模型分数都应该拿它当 0 点, 而不是拿 0.0 当 0 点。
    h = len(real_F) // 2
    _, floor = score_all(real_F[:h], real_F[h:])
    # (b) GAUSSIAN: 直接从 N(0,Σ) 采样, 不训练。这是"只有二阶统计量、没有任何高阶结构"
    #     的水平, 也正是 HDGN arm 的采样起点 x_T。它标出训练到底带来了多少增益。
    gsrc = NoiseSource(D, Sigma, device=args.device)
    gz = gsrc.sample(min(args.n_gen, 4000)).cpu().numpy().reshape(-1, args.T, 43) * sd + mu
    _, gauss = score_all(real_F, gz)
    results["baselines"] = {"floor_real_vs_real": floor, "gaussian_sigma_untrained": gauss}
    log(f"[base] FLOOR(真实vs真实)  l1={floor['l1']:.4f} ws={floor['wasserstein']:.4f} ks={floor['ks']:.4f}")
    log(f"[base] N(0,Σ) 未训练      l1={gauss['l1']:.4f} ws={gauss['wasserstein']:.4f} ks={gauss['ks']:.4f}")

    for arm in ("iid", "hdgn"):
        noise = NoiseSource(D, None if arm == "iid" else Sigma, device=args.device)

        def eval_fn(model, noise, ab, step):
            model.eval()
            t0 = time.time()
            with torch.no_grad():
                g = ddim_sample(model, noise, ab, args.n_gen, D, args.device,
                                nfe=args.eval_nfe, seed=1234)
            gen_F = g.float().cpu().numpy().reshape(-1, args.T, 43) * sd + mu
            per, means = score_all(real_F, gen_F)
            model.train()
            return {**means, "sample_s": time.time() - t0, "_per": per}

        log(f"[train] arm={arm}")
        model, hist, evals, wall = train_arm(
            Xtr, noise, args.steps, D, args.device, seed=args.seed,
            hidden=args.hidden, layers=args.layers, batch=args.batch, lr=args.lr,
            eval_at=set(eval_at), eval_fn=eval_fn, log=log)
        log(f"[train] arm={arm} 用时 {wall:.1f}s")

        # NFE 扫描(只在最终模型上)
        ab, _ = cosine_alphas(1000); ab = ab.to(args.device)
        nfe_res = {}
        for nfe in [int(x) for x in args.nfe_sweep.split(",")]:
            model.eval()
            t0 = time.time()
            with torch.no_grad():
                g = ddim_sample(model, noise, ab, args.n_gen, D, args.device, nfe=nfe, seed=1234)
            gen_F = g.float().cpu().numpy().reshape(-1, args.T, 43) * sd + mu
            per, means = score_all(real_F, gen_F)
            nfe_res[nfe] = {**means, "sample_s": time.time() - t0}
            log(f"    [{arm}] NFE={nfe:4d}  l1={means['l1']:.4f} ws={means['wasserstein']:.4f} ks={means['ks']:.4f}")

        results["arms"][arm] = {
            "loss_hist": hist, "wall_s": wall,
            "evals": {str(k): v for k, v in evals.items()},
            "nfe_sweep": {str(k): v for k, v in nfe_res.items()},
        }
        torch.save(model.state_dict(), os.path.join(args.out, f"model_{arm}.pt"))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"[done] -> {os.path.join(args.out,'results.json')}")


if __name__ == "__main__":
    main()
