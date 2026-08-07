#!/usr/bin/env python3
"""噪声协方差 diffusion 在 LOB 数据上的四臂受控实验。

方法与损失直接取自论文 codebase (OpenReviewAnonymous/Diffusion, NeurIPS 2025):
  - 模型主干与 AdaptiveDDPM 相同: Embedding(t) 32 维, [input_dim+32]->H->H->[input_dim]
  - 可学 Cholesky 因子 L: 下三角 + 对角 softplus, 噪声 ε = ξ Lᵀ, ξ~N(0,I)
  - 损失 = ‖L⁻¹(ε_θ-ε)‖²  (Mahalanobis, 非 MSE) + λ1‖L‖_F² + λ2‖diag L‖²

四个臂
------
  iid          Σ=I,          L 不学,  主干 H=256      —— 论文的 DDPM baseline (任务 1)
  iid_wide     Σ=I,          L 不学,  主干加宽到与 hdgn_learned 同总参数
  hdgn_fixed   Σ=样本协方差, L 不学,  主干 H=256      —— 零额外参数, 与 iid 同容量
  hdgn_learned Σ=LLᵀ,        L 可学,  主干 H=256      —— 论文方法 (任务 2)

为什么必须有 iid_wide 和 hdgn_fixed: L 占 473,344 参数(总量的 52.5%)。只比
iid(428k) 和 hdgn_learned(901k) 的话, 分不清赢的是协方差还是参数量。
hdgn_fixed 用和 iid 一样的参数量拿到正确的协方差, iid_wide 用和 hdgn_learned
一样的参数量但没有协方差 —— 两个臂各自隔离一个变量。

统一损失让四臂同尺度可比: ‖L⁻¹ε‖² 对任何 Σ 都服从 χ²_D, 期望恒为 D。
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, noise_covariance
from normalize import ChannelNormalizer
from lobbench_eval import score_all, COVERED, NOT_COVERED

DS = "/lus/lfs1aip2/projects/public/u6gb/datasets/lob_flat43_example_20260807T135612Z"
T8 = ["GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"]


class Arm(nn.Module):
    """论文 AdaptiveDDPM 的结构, 把隐层宽度与 L 是否可学参数化出来。"""

    def __init__(self, T, D, hidden=256, learn_L=False, L_init=None, emb=32,
                 trunk="paper", depth=2):
        super().__init__()
        self.T, self.D, self.input_dim = T, D, T * D
        self.time_embed = nn.Embedding(1000, emb)
        self.trunk = trunk
        if trunk == "paper":
            # 论文 AdaptiveDDPM 原样: 两层 ReLU MLP, 无归一化无残差
            self.model = nn.Sequential(
                nn.Linear(self.input_dim + emb, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, self.input_dim))
        else:
            # 加深版: 残差 + LayerNorm + SiLU。论文原配置是为 input_dim=200 设计的,
            # 本数据 input_dim=688, 原主干会把四个臂一起压在饱和区而失去区分度。
            # 主干对四臂完全相同, 因此不影响"唯一变量是 Σ"的对照。
            self.inp = nn.Linear(self.input_dim + emb, hidden)
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden)) for _ in range(depth)])
            self.outp = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.input_dim))
        self.learn_L = learn_L
        self.register_buffer("tril_mask", torch.tril(torch.ones(self.input_dim, self.input_dim)))
        if learn_L:
            self.L_params = nn.Parameter(torch.randn(self.input_dim, self.input_dim) * 0.01)
            self.register_buffer("L_fixed", torch.empty(0))
        else:
            self.register_parameter("L_params", None)
            L = torch.eye(self.input_dim) if L_init is None else torch.as_tensor(L_init, dtype=torch.float32)
            self.register_buffer("L_fixed", L)

    def get_L(self):
        if not self.learn_L:
            return self.L_fixed
        L = self.L_params * self.tril_mask                     # 论文原式
        idx = torch.arange(self.input_dim, device=L.device)
        L = L.clone()
        L[idx, idx] = F.softplus(L[idx, idx])
        return L

    def get_noise(self, n, generator=None):
        xi = torch.randn(n, self.input_dim, device=self.tril_mask.device, generator=generator)
        return xi @ self.get_L().T

    def forward(self, x_flat, t):
        h = torch.cat([x_flat, self.time_embed(t)], 1)
        if self.trunk == "paper":
            return self.model(h)
        h = self.inp(h)
        for b in self.blocks:
            h = h + b(h)
        return self.outp(h)


def cosine_alphas(Tsteps, s=0.008, device="cpu"):
    x = torch.linspace(0, Tsteps, Tsteps + 1)
    f = torch.cos(((x / Tsteps) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = torch.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    return torch.cumprod(1 - betas, 0).to(device)


def mahalanobis_loss(L, r):
    """论文的 recon loss: z = L⁻¹ rᵀ, 返回 mean(sum(z², dim=1)) / D 使其无量纲(≈1)。"""
    z = torch.linalg.solve_triangular(L, r.T, upper=False).T
    return (z ** 2).sum(1).mean() / r.shape[1]


@torch.no_grad()
def ddim_sample(model, ab, n, nfe, device, seed=0, Tsteps=1000, x0_clip=None):
    """DDIM 采样。

    x0_clip 不是可选的美化项: cosine schedule 令 ᾱ_T -> 0 (实测 ab[999]=2.43e-9),
    而 x0 = (x - sqrt(1-ᾱ)ε_θ)/sqrt(ᾱ) 会把 ε_θ 的误差放大 1/sqrt(ᾱ) = 20,291 倍。
    没有 clip 时, 即使模型把噪声预测得很准(loss 0.15, 解释 85% 方差), x0 也会在
    第一步爆炸, 整条链作废 —— 表现为 loss 很低但生成质量恒定为零。
    这就是 DDPM/DDIM 参考实现里的 clip_denoised。
    """
    g = torch.Generator(device=device); g.manual_seed(seed)
    x = model.get_noise(n, generator=g)
    ts = torch.linspace(Tsteps - 1, 0, nfe).long().to(device)
    for i in range(nfe):
        t = ts[i]; a_t = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < nfe else torch.tensor(1.0, device=device)
        e = model(x, t.repeat(n))
        x0 = (x - (1 - a_t).sqrt() * e) / a_t.sqrt()
        if x0_clip is not None:
            x0 = x0.clamp(-x0_clip, x0_clip)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--trunk", default="paper", choices=["paper","deep"])
    ap.add_argument("--norm", default="quantile", choices=["zscore","quantile"])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam1", type=float, default=1e-2)   # 论文 HP_LAMBDA_1 默认
    ap.add_argument("--lam2", type=float, default=0.0)    # 论文 HP_LAMBDA_2 默认
    ap.add_argument("--n-gen", type=int, default=4000)
    ap.add_argument("--eval-nfe", type=int, default=50)
    ap.add_argument("--nfe-sweep", default="5,10,20,50,100,200")
    ap.add_argument("--arms", default="iid,iid_wide,hdgn_fixed,hdgn_learned")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = lambda *a: print(*a, flush=True)

    dev = args.device
    X = load_dataset(DS, T8, T=args.T, stride=args.T, max_rows_per_ticker=args.rows, seed=args.seed)
    D = args.T * 43
    ntr_w = int(0.9 * len(X))
    normer = ChannelNormalizer(args.norm).fit(X[:ntr_w])     # 只在训练片上 fit, 防泄漏
    Xz = normer.transform(X)
    Xf = Xz.reshape(len(Xz), D).astype(np.float32)
    ntr = ntr_w
    Xtr, Xte = Xf[:ntr], Xf[ntr:]
    Sigma = noise_covariance(Xtr)
    evals_w = np.linalg.eigvalsh(Sigma)[::-1]
    cond = float(evals_w[0] / max(evals_w[-1], 1e-12))
    L_data = np.linalg.cholesky(Sigma).astype(np.float32)
    real_F = normer.inverse_transform(Xte.reshape(-1, args.T, 43))
    # x0 clip 阈值: 取训练数据的 99.99% 分位数再留 20% 余量, 保证不裁掉真实数据
    x0_clip = float(np.quantile(np.abs(Xtr), 0.9999) * 1.2)
    log(f"[setup] dev={dev} 窗口{X.shape} D={D} train={len(Xtr)} test={len(Xte)} Σcond={cond:.3e}")
    log(f"[setup] x0_clip={x0_clip:.2f} (训练数据 |z| 的 max={np.abs(Xtr).max():.2f})")

    # 基线: FLOOR(真实vs真实) 与 未训练的 N(0,Σ)
    h = len(real_F) // 2
    _, floor = score_all(real_F[:h], real_F[h:])
    gz = normer.inverse_transform((torch.randn(min(args.n_gen,4000), D) @ torch.as_tensor(L_data).T).numpy().reshape(-1, args.T, 43))
    _, gauss = score_all(real_F, gz)
    log(f"[base] FLOOR 真实vs真实   l1={floor['l1']:.4f} ws={floor['wasserstein']:.4f} ks={floor['ks']:.4f}")
    log(f"[base] N(0,Σ) 未训练      l1={gauss['l1']:.4f} ws={gauss['wasserstein']:.4f} ks={gauss['ks']:.4f}")

    # iid_wide 的宽度: 实际构造模型数参数, 令其总量 ≈ hdgn_learned (主干 + L 的 D² 个参数)
    def n_params_of(H):
        m = Arm(args.T, 43, hidden=H, learn_L=False, L_init=None,
                trunk=args.trunk, depth=args.depth)
        return sum(p.numel() for p in m.parameters())
    target = n_params_of(args.hidden) + D * D
    Hw = args.hidden
    while n_params_of(Hw) < target:
        Hw += 16
    log(f"[setup] trunk={args.trunk} depth={args.depth} hidden={args.hidden}; "
        f"iid_wide 宽度 {Hw} (目标 {target:,}, 实得 {n_params_of(Hw):,})")

    eval_at = sorted({int(round(args.steps * f)) for f in
                      (0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.5, 0.72, 1.0)})
    log(f"[setup] 评估点 {eval_at}")

    ab = cosine_alphas(1000, device=dev)
    results = {"config": vars(args), "D": D, "sigma_cond": cond,
               "n_train": len(Xtr), "n_test": len(Xte), "hidden_wide": Hw, "x0_clip": x0_clip, "norm": args.norm,
               "features_covered": list(COVERED), "features_not_covered": list(NOT_COVERED),
               "baselines": {"floor_real_vs_real": floor, "gaussian_sigma_untrained": gauss},
               "arms": {}}

    TK = dict(trunk=args.trunk, depth=args.depth)
    SPEC = {
        "iid":          dict(hidden=args.hidden, learn_L=False, L_init=None, **TK),
        "iid_wide":     dict(hidden=Hw,          learn_L=False, L_init=None, **TK),
        "hdgn_fixed":   dict(hidden=args.hidden, learn_L=False, L_init=L_data, **TK),
        "hdgn_learned": dict(hidden=args.hidden, learn_L=True,  L_init=None, **TK),
    }

    Xtr_t = torch.as_tensor(Xtr, device=dev)
    for arm in args.arms.split(","):
        torch.manual_seed(args.seed)                        # 四臂同初始化种子
        model = Arm(args.T, 43, **SPEC[arm]).to(dev)
        npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        gen = torch.Generator(device=dev); gen.manual_seed(args.seed + 1)   # 同数据顺序
        log(f"[train] arm={arm} 参数={npar:,}")
        hist, evs = [], {}
        t0 = time.time()
        for step in range(1, args.steps + 1):
            idx = torch.randint(0, len(Xtr_t), (args.batch,), device=dev, generator=gen)
            x0 = Xtr_t[idx]
            t = torch.randint(0, 1000, (args.batch,), device=dev, generator=gen)
            L = model.get_L()
            eps = torch.randn(args.batch, D, device=dev, generator=gen) @ L.T
            a = ab[t][:, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            r = model(xt, t) - eps
            loss = mahalanobis_loss(L, r)
            if model.learn_L:
                loss = loss + args.lam1 * (L ** 2).sum() + args.lam2 * (torch.diag(L) ** 2).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 100 == 0:
                hist.append((step, loss.detach().item()))
            if step in eval_at:
                model.eval()
                g = ddim_sample(model, ab, args.n_gen, args.eval_nfe, dev, seed=1234, x0_clip=x0_clip)
                gen_F = normer.inverse_transform(g.float().cpu().numpy().reshape(-1, args.T, 43))
                per, means = score_all(real_F, gen_F)
                evs[step] = {**means, "_per": per}
                model.train()
                log(f"    [{arm}] step {step:6d} loss {loss.item():.4f} "
                    f"l1={means['l1']:.4f} ws={means['wasserstein']:.4f} ks={means['ks']:.4f}")
        wall = time.time() - t0
        nfe_res = {}
        model.eval()
        for nfe in [int(x) for x in args.nfe_sweep.split(",")]:
            ts = time.time()
            g = ddim_sample(model, ab, args.n_gen, nfe, dev, seed=1234, x0_clip=x0_clip)
            gen_F = normer.inverse_transform(g.float().cpu().numpy().reshape(-1, args.T, 43))
            _, means = score_all(real_F, gen_F)
            nfe_res[str(nfe)] = {**means, "sample_s": time.time() - ts}
            log(f"    [{arm}] NFE={nfe:4d} l1={means['l1']:.4f} ws={means['wasserstein']:.4f} ks={means['ks']:.4f}")
        results["arms"][arm] = {"n_params": npar, "wall_s": wall, "loss_hist": hist,
                                "evals": {str(k): v for k, v in evs.items()}, "nfe_sweep": nfe_res}
        torch.save(model.state_dict(), os.path.join(args.out, f"model_{arm}.pt"))
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log(f"[train] arm={arm} 完成, 用时 {wall:.1f}s")
    log(f"[done] -> {os.path.join(args.out,'results.json')}")


if __name__ == "__main__":
    main()
