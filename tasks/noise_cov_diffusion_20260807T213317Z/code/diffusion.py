#!/usr/bin/env python3
"""噪声协方差 diffusion: iid (Σ=I) vs HDGN (Σ=数据协方差) 的受控对照。

对照设计
--------
两个 arm 共享: 模型结构、参数初始化种子、数据、数据顺序、优化器、学习率、β 调度、
评估用的随机种子。**唯一变量是前向噪声的协方差 Σ**。
两个 Σ 都归一化到 trace(Σ)=D, 所以注入的总噪声能量相同 —— 比较的是噪声的
"形状"而不是"强度"。

前向:   x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε,  ε ~ N(0, Σ),  ε = L z, z~N(0,I)
损失:   || ε_pred - ε ||²   (两 arm 同尺度可比, 因 trace 相同)
采样:   DDIM, 可变步数 -> 用来测 NFE 维度的成本
"""
from __future__ import annotations

import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 模型
class Denoiser(nn.Module):
    """MLP 去噪器。时间用 sinusoidal 嵌入后与输入拼接。"""

    def __init__(self, D, hidden=1024, layers=4, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        blocks, d_in = [], D
        for _ in range(layers):
            blocks.append(nn.Linear(d_in, hidden)); d_in = hidden
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, D)
        self.act = nn.SiLU()

    def temb(self, t):
        half = self.tdim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward(self, x, t):
        h_t = self.tmlp(self.temb(t))
        h = x
        for i, (blk, nrm) in enumerate(zip(self.blocks, self.norms)):
            h = blk(h)
            h = h + h_t if i == 0 else h
            h = self.act(nrm(h))
        return self.out(h)


# ---------------------------------------------------------------- 调度
def cosine_alphas(Tsteps, s=0.008):
    x = torch.linspace(0, Tsteps, Tsteps + 1)
    f = torch.cos(((x / Tsteps) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = torch.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    return torch.cumprod(1 - betas, 0), betas


# ---------------------------------------------------------------- 噪声源
class NoiseSource:
    """按协方差 Σ 采样噪声。iid 时 L=I, 直接返回标准正态(省一次矩阵乘)。"""

    def __init__(self, D, Sigma=None, device="cpu", dtype=torch.float32):
        self.D, self.device = D, device
        if Sigma is None:
            self.L, self.kind = None, "iid"
        else:
            # Cholesky; 若数值上非正定则用特征分解兜底
            S = torch.as_tensor(Sigma, dtype=torch.float64)
            try:
                L = torch.linalg.cholesky(S)
            except Exception:
                w, V = torch.linalg.eigh(S)
                L = V @ torch.diag(torch.clip(w, 1e-10).sqrt())
            self.L = L.to(device=device, dtype=dtype)
            self.kind = "hdgn"

    def sample(self, n, generator=None):
        z = torch.randn(n, self.D, device=self.device, generator=generator)
        return z if self.L is None else z @ self.L.T


# ---------------------------------------------------------------- 训练
def train_arm(X, noise, steps, D, device, seed=0, hidden=1024, layers=4,
              batch=256, lr=2e-4, Tsteps=1000, eval_at=(), eval_fn=None, log=print):
    torch.manual_seed(seed)                       # 两 arm 同初始化
    model = Denoiser(D, hidden, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ab, _ = cosine_alphas(Tsteps)
    ab = ab.to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = len(Xt)
    gen = torch.Generator(device=device); gen.manual_seed(seed + 1)   # 两 arm 同数据顺序
    hist, evals = [], {}
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), device=device, generator=gen)
        x0 = Xt[idx]
        t = torch.randint(0, Tsteps, (batch,), device=device, generator=gen)
        eps = noise.sample(batch, generator=gen)
        a = ab[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = ((model(xt, t) - eps) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0:
            hist.append((step, loss.detach().item()))
        if step in eval_at and eval_fn is not None:
            evals[step] = eval_fn(model, noise, ab, step)
            log(f"    [{noise.kind}] step {step:6d}  loss {float(loss):.4f}  "
                f"{ {k: round(v,4) for k,v in evals[step].items() if isinstance(v,float)} }")
    return model, hist, evals, time.time() - t0


# ---------------------------------------------------------------- 采样
@torch.no_grad()
def ddim_sample(model, noise, ab, n, D, device, nfe=50, seed=0, Tsteps=1000):
    """DDIM 确定性采样。噪声协方差通过初始点 x_T ~ N(0,Σ) 进入。"""
    g = torch.Generator(device=device); g.manual_seed(seed)
    x = noise.sample(n, generator=g)
    ts = torch.linspace(Tsteps - 1, 0, nfe).long().to(device)
    for i in range(nfe):
        t = ts[i]
        a_t = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < nfe else torch.tensor(1.0, device=device)
        e = model(x, t.repeat(n))
        x0 = (x - (1 - a_t).sqrt() * e) / a_t.sqrt()
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x
