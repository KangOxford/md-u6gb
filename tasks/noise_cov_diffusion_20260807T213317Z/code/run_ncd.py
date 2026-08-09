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
from data import (load_dataset, noise_covariance, noise_covariance_toeplitz,
                  DS, T8)   # DS/T8 再导出, 保持既有调用点
from normalize import ChannelNormalizer
from lobbench_eval import score_all, COVERED, NOT_COVERED



class Arm(nn.Module):
    """论文 AdaptiveDDPM 的结构, 把隐层宽度与 L 是否可学参数化出来。"""

    def __init__(self, T, D, hidden=256, learn_L=False, L_init=None, emb=32,
                 trunk="paper", depth=2, L_bank=None, norm_L=False, anchor=None,
                 param_mode="chol", logdet=False, eig=None, tbins=8, tdep_pow=1.0):
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
        self.norm_L = norm_L
        # param_mode 决定 L 的参数化方式(见 get_L):
        #   "chol"     论文原式(+ v2/v3 的变体): 下三角矩阵, 对角过 softplus
        #   "logchol"  L = tril(A,-1) + diag(exp(d)) —— 正定性由构造保证, logdet 解析
        #   "spectral" L = Q diag(exp(s)), Q 为 Sigma_data 的特征向量(冻结)
        # logdet=True 时在 loss 里补回高斯 NLL 的 logdet(Sigma) 项。
        # 根因见 REPORT 十三H: 论文的 tr(Sigma^-1 C) 在 trace 约束下的最优解是 C^{1/2},
        # 开方本身就把谱推向各向同性; 补回 logdet 后最优解变成 C, 且平凡解自动被堵死。
        self.param_mode = param_mode
        self.use_logdet = logdet
        # anchor: 把可学的 L 锚定在 Sigma_data 的 Cholesky 上。
        # 动机(实测): 固定的样本协方差精确编码了数据的时间相关结构; 而自由学习的 L 在
        # 优化 Mahalanobis loss —— 那个目标不奖励时间依赖, 梯度会把 L 推向"让去噪更容易"
        # 的方向, 途中把时间结构丢掉(v2 的 acf_lag1 比 hdgn_fixed 差 2.4-27 倍)。
        # 故改为学习**受约束的偏离**: L = L_anchor + dL, 并惩罚 ||dL||_F。
        if anchor is not None:
            La = torch.as_tensor(anchor, dtype=torch.float32)
            self.register_buffer("L_anchor", La)
            # logdet(Sigma_a) = 2*sum(log diag(L_a)) —— KL 项要用, 解析且是常数
            self.register_buffer("logdet_anchor",
                                 2.0 * torch.log(torch.diagonal(La).clamp_min(1e-12)).sum())
        else:
            self.register_buffer("L_anchor", torch.empty(0))
            self.register_buffer("logdet_anchor", torch.zeros(()))
        # L_bank: (K, D, D) —— 每个市场 regime 一个 Cholesky 因子(dynamic 臂)
        self.has_bank = L_bank is not None
        if self.has_bank:
            self.register_buffer("L_bank", torch.as_tensor(L_bank, dtype=torch.float32))
        self.register_buffer("tril_mask", torch.tril(torch.ones(self.input_dim, self.input_dim)))
        if learn_L and param_mode == "logchol":
            # v4/v5: L = tril(A,-1) + diag(exp(d))。起点精确等于 L_init(默认 Sigma_data)。
            Lm = torch.as_tensor(L_init, dtype=torch.float32).clone()
            self.register_buffer("strict_mask",
                                 torch.tril(torch.ones(self.input_dim, self.input_dim), -1))
            self.A = nn.Parameter(Lm * self.strict_mask)
            self.dlog = nn.Parameter(torch.log(torch.diagonal(Lm).clamp_min(1e-8)))
            self.register_buffer("L_fixed", torch.empty(0))
            self.register_parameter("L_params", None)
        elif learn_L and param_mode == "tdep":
            # v8: **随 t 变形的协方差**。Σ(t) = Σ_data + δ(t)·Δ, δ(t)=ᾱ_t^tdep_pow。
            #
            # 为什么这样能同时保住守恒并拿到偏离的收益:
            #   Cov(x_t) = ᾱ_t Σ_data + (1-ᾱ_t) Σ(t)
            # 偏离项被 (1-ᾱ_t) 加权, 而 δ(t)=ᾱ_t 在小 t 才大 —— 两者相乘 ᾱ_t(1-ᾱ_t) ≤ 1/4,
            # 且**两端都趋于零**。大 t(采样开头、跳步最大处)精确等于 Σ_data。
            # 免费的好处: DDIM 只在起点抽一次噪声, 而 ᾱ_T=2.43e-9 → δ(T)≈0,
            # 所以**采样先验精确等于 Σ_data**, 与 hdgn_fixed 一致。
            #
            # 实现上复用 hdgn_regime 那套 L_bank 分组三角求解, 只是把索引从"市场态"
            # 换成"t 分箱" —— 不新写一套(任务 9)。
            Lm = torch.as_tensor(L_init, dtype=torch.float32).clone()
            self.register_buffer("strict_mask",
                                 torch.tril(torch.ones(self.input_dim, self.input_dim), -1))
            self.register_buffer("L_base_off", Lm * self.strict_mask)
            self.register_buffer("dlog_base", torch.log(torch.diagonal(Lm).clamp_min(1e-8)))
            self.A = nn.Parameter(torch.zeros(self.input_dim, self.input_dim))
            self.dlog = nn.Parameter(torch.zeros(self.input_dim))
            self.n_tbins = int(tbins)
            self.tdep_pow = float(tdep_pow)
            # δ_k = 该分箱内 ᾱ_t 的均值的 tdep_pow 次方。用均值而非中点更平滑,
            # 且 cosine schedule 下 ᾱ 在两端极不均匀(ab[999]=2.43e-9)。
            _ab = cosine_alphas(1000)
            _e = torch.linspace(0, 1000, self.n_tbins + 1).long()
            self.register_buffer("delta_k", torch.stack(
                [_ab[_e[k]:_e[k + 1]].mean() ** self.tdep_pow for k in range(self.n_tbins)]))
            self.has_bank = True                       # 走 bank 的分组求解路径
            self.register_buffer("L_fixed", torch.empty(0))
            self.register_parameter("L_params", None)
        elif learn_L and param_mode == "spectral":
            # v6: 只学 Sigma_data 特征基里的 D 个特征值, 特征向量冻结。
            # 依据(诊断 13H.1): 自由学习的漂移 99.9% 都在谱上, 旋转只占 0.1%,
            # 所以谱族足以表达它想学的东西, 而参数量少 344 倍且时间结构由构造保住。
            lam, Qm = eig
            self.register_buffer("Q", torch.as_tensor(Qm, dtype=torch.float32))
            self.s = nn.Parameter(0.5 * torch.log(
                torch.as_tensor(lam, dtype=torch.float32).clamp_min(1e-12)))
            self.register_buffer("L_fixed", torch.empty(0))
            self.register_parameter("L_params", None)
        elif learn_L:
            if L_init is None:
                # 论文原版: randn*0.01 -> 非对角≈0, 对角 softplus(0)=0.693,
                # 即初始 Sigma≈0.48*I —— 从一个近似 iid 的噪声起步, 要用 47 万个参数
                # 从零爬到样本协方差。这是 hdgn_learned 表现差的主因之一。
                init = torch.randn(self.input_dim, self.input_dim) * 0.01
            else:
                # 用样本协方差的 Cholesky 初始化: 起点就是 hdgn_fixed 的终点,
                # 学习只需在此基础上微调。对角要反解 softplus: x = log(exp(y)-1)。
                Lm = torch.as_tensor(L_init, dtype=torch.float32).clone()
                d = torch.diagonal(Lm).clamp_min(1e-6)
                init = Lm
                torch.diagonal(init).copy_(torch.log(torch.expm1(d).clamp_min(1e-8)))
            if anchor is not None:
                init = torch.zeros(self.input_dim, self.input_dim)   # dL 从 0 起, 即起点 = Sigma_data
            self.L_params = nn.Parameter(init)
            self.register_buffer("L_fixed", torch.empty(0))
        else:
            self.register_parameter("L_params", None)
            L = torch.eye(self.input_dim) if L_init is None else torch.as_tensor(L_init, dtype=torch.float32)
            self.register_buffer("L_fixed", L)

    def _norm_scale(self, L):
        """trace(Sigma)=||L||_F^2 归一化到 D, 返回 (缩放后的 L, log 缩放系数)。
        logdet 要用同一个系数修正, 故这里一并返回而不是各算各的。"""
        if not self.norm_L:
            return L, L.new_zeros(())
        c = torch.sqrt(self.input_dim / ((L ** 2).sum() + 1e-12))
        return L * c, torch.log(c)

    def get_L(self):
        if not self.learn_L:
            return self.L_fixed
        if self.param_mode == "logchol":
            L = self.A * self.strict_mask + torch.diag(torch.exp(self.dlog))
            return self._norm_scale(L)[0]
        if self.param_mode == "spectral":
            return self._norm_scale(self.Q * torch.exp(self.s))[0]
        if self.param_mode == "tdep":
            # tdep 没有单一的 L(它随 t 变)。返回偏离最大的那一箱作代表,
            # 供诊断与"学到的 Σ 有多偏"这类问题使用; 训练/采样一律走 bank()。
            return self.bank()[int(self.delta_k.argmax())]
        if self.L_anchor.numel() > 0:
            # 锚定模式: L_params 直接作为增量 dL(初始化为 0), 不过 softplus
            L = self.L_anchor + self.L_params * self.tril_mask
        else:
            L = self.L_params * self.tril_mask                 # 论文原式
        if self.L_anchor.numel() == 0:
            idx = torch.arange(self.input_dim, device=L.device)
            L = L.clone()
            L[idx, idx] = F.softplus(L[idx, idx])
        if self.norm_L:
            # trace(Sigma) = trace(LL^T) = ||L||_F^2, 归一化到 = D。
            # 两个作用: (a) 与 hdgn_fixed/hdgn_regime 注入等量噪声能量, 对比才公平;
            # (b) 直接堵死 "L->inf 让 ||L^-1 r||^2 -> 0" 的平凡解 —— 论文用 Frobenius
            #     正则堵同一个洞, 但归一化不扭曲优化目标, 且有了它 lam1 可以设 0。
            L = L * torch.sqrt(self.input_dim / ((L ** 2).sum() + 1e-12))
        return L

    def bank(self):
        """返回 (K, D, D) 的 L 列表。tdep 臂是**可微**的动态 bank, regime 臂是静态 buffer。

        tdep: L_k = tril(L_data,-1) + δ_k·A ⊙ mask  +  diag(exp(dlog_data + δ_k·dlog))
        δ_k=0 时精确等于 L_data(对角走 exp 保证正定, 无需投影)。
        """
        if self.param_mode != "tdep":
            return self.L_bank
        d = self.delta_k.view(-1, 1, 1)
        off = self.L_base_off.unsqueeze(0) + d * (self.A * self.strict_mask).unsqueeze(0)
        dia = torch.exp(self.dlog_base.unsqueeze(0) + self.delta_k.view(-1, 1) * self.dlog)
        return off + torch.diag_embed(dia)

    def tbin_of(self, t):
        """t(0..999) → 分箱索引, 与 delta_k 对齐。"""
        return (t.long() * self.n_tbins // 1000).clamp(0, self.n_tbins - 1)

    def get_noise(self, n, generator=None, regimes=None):
        xi = torch.randn(n, self.input_dim, device=self.tril_mask.device, generator=generator)
        if not self.has_bank:
            return xi @ self.get_L().T
        # 每个样本用它所属 regime 的 L: 按 regime 分组做矩阵乘, K 次而非 n 次
        Lb = self.bank()
        out = torch.empty_like(xi)
        for k in range(Lb.shape[0]):
            m = regimes == k
            if m.any():
                out[m] = xi[m] @ Lb[k].T
        return out

    def logdet_over_D(self):
        """返回 logdet(Sigma)/D。解析式, 不做行列式分解。

        补这一项的理由见 REPORT 十三H: 论文的 recon loss 是高斯 NLL 去掉归一化常数
        (1/2)logdet(Sigma) 的结果。Sigma 固定时那是常数, 去掉无碍; 一旦 Sigma 变成
        待学参数, 去掉它就让目标退化 —— 最优解从 C 变成 C^{1/2}, 而开方把谱推向各向同性。
        """
        if self.param_mode == "logchol":
            L = self.A * self.strict_mask + torch.diag(torch.exp(self.dlog))
            raw = 2.0 * self.dlog.sum()
        elif self.param_mode == "spectral":
            L = self.Q * torch.exp(self.s)
            raw = 2.0 * self.s.sum()
        else:                                   # 论文原式: 只能走三角阵对角
            L = self.get_L()
            return 2.0 * torch.log(torch.diagonal(L).clamp_min(1e-12)).sum() / self.input_dim
        _, logc = self._norm_scale(L)
        return (raw + 2.0 * self.input_dim * logc) / self.input_dim

    def kl_to_anchor(self):
        """KL( N(0,Σ_anchor) ‖ N(0,Σ) ) / D —— 一个**外生**的锚定项。

        为什么需要外生项(见 REPORT 13H.4c): 去噪损失 tr(Σ⁻¹C) 里 C 内生于 Σ,
        `C ≈ ρ²Σ` 使该项退化成常数 ρ²D, 对 Σ 不再提供任何信号。此时无论补
        logdet 还是加 Frobenius 正则, 都只是在换"被打开的那扇门"。
        唯一的出路是给 Σ 一个不随它自身缩放的信号, Σ_data 就是。

        为什么用 KL 而不是 v3 的 ‖ΔL‖²_F: Frobenius 惩罚被**大元素主导**,
        等于把估得最准的方向钉死、放任估得最差的小特征值方向乱跑 —— 正好搞反。
        KL 惩罚的是对数尺度上的相对偏离, 对所有特征方向一视同仁。

            KL = ½[ tr(Σ⁻¹Σ_a) − D + logdet Σ − logdet Σ_a ],
            tr(Σ⁻¹Σ_a) = ‖L⁻¹L_a‖²_F
        """
        La = self.L_anchor
        Lc = self.bank()[int(self.delta_k.argmax())] if self.param_mode == "tdep" else self.get_L()
        Z = torch.linalg.solve_triangular(Lc, La, upper=False)
        tr = (Z ** 2).sum()
        if self.param_mode == "tdep":
            ld = 2.0 * (self.dlog_base + self.delta_k.max() * self.dlog).sum()
            return 0.5 * (tr / self.input_dim - 1.0
                          + (ld - self.logdet_anchor) / self.input_dim)
        return 0.5 * (tr / self.input_dim - 1.0
                      + self.logdet_over_D() - self.logdet_anchor / self.input_dim)

    def whiten(self, r):
        """z = L^-1 r(按行)。谱族有解析逆, 不必解三角系统 —— 这一条同时让推理更快。"""
        if self.param_mode == "spectral":
            # L = c·Q diag(e^s)  =>  L^-1 = (1/c) diag(e^-s) Q^T  =>  z = (r@Q)·e^-s / c
            _, logc = self._norm_scale(self.Q * torch.exp(self.s))
            return (r @ self.Q) * torch.exp(-self.s - logc)
        return torch.linalg.solve_triangular(self.get_L(), r.T, upper=False).T

    def mahalanobis(self, r, regimes=None):
        """按 regime 分组解三角系统, 返回 ||L^-1 r||^2 / D。"""
        if not self.has_bank:
            z = self.whiten(r)
            return (z ** 2).sum(1).mean() / r.shape[1]
        Lb = self.bank()          # tdep 臂是可微的动态 bank, regime 臂是静态 buffer
        tot = 0.0
        for k in range(Lb.shape[0]):
            m = regimes == k
            if m.any():
                z = torch.linalg.solve_triangular(Lb[k], r[m].T, upper=False).T
                tot = tot + (z ** 2).sum(1).sum()
        return tot / (r.shape[0] * r.shape[1])

    def forward(self, x_flat, t):
        h = torch.cat([x_flat, self.time_embed(t)], 1)
        if self.trunk == "paper":
            return self.model(h)
        h = self.inp(h)
        for b in self.blocks:
            h = h + b(h)
        return self.outp(h)


def arm_specs(hidden, hidden_wide, L_data, L_bank=None, eig=None,
              L_toe=None, trunk="deep", depth=3):
    """所有臂的定义 —— 训练(run_ncd)与重评估(reeval_ts)共用的**单一来源**。

    这份定义曾经在两个文件里各写一遍。后果不是报错而是静默漏做: reeval 的循环是
    `for arm in SPEC`, 新增的臂只要没同步过去就直接不在集合里, 不会有任何提示。
    与 lobbench_eval/feature_ts_metrics 的特征定义重复是同一个病 ——
    **能被遗忘的重复, 迟早会被遗忘。**
    """
    TK = dict(trunk=trunk, depth=depth)
    S = {
        "iid":            dict(hidden=hidden,      learn_L=False, L_init=None, **TK),
        "iid_wide":       dict(hidden=hidden_wide, learn_L=False, L_init=None, **TK),
        "hdgn_fixed":     dict(hidden=hidden,      learn_L=False, L_init=L_data, **TK),
        "hdgn_learned":   dict(hidden=hidden,      learn_L=True,  L_init=None, **TK),
        # 修复版: 样本协方差初始化 + trace 归一化(故 lam1 可设 0)
        "hdgn_learned_v2":   dict(hidden=hidden, learn_L=True, L_init=L_data,
                                  norm_L=True, **TK),
        # 消融: 只加 trace 归一化, 仍随机初始化 —— 用来分离两项修改各自的贡献
        "hdgn_learned_norm": dict(hidden=hidden, learn_L=True, L_init=None,
                                  norm_L=True, **TK),
        # v3: 锚定学习 —— 起点即 Sigma_data, 只学受惩罚的偏离 dL
        "hdgn_learned_v3":   dict(hidden=hidden, learn_L=True, L_init=None,
                                  norm_L=True, anchor=L_data, **TK),
        # ---- 任务 19: 补回 logdet 后的三个臂(理论根因见 REPORT 十三H) ----
        # v4: log-Cholesky + logdet, 无 trace 归一化 —— 让 Sigma 自由收敛到网络误差协方差 C
        "hdgn_learned_v4":   dict(hidden=hidden, learn_L=True, L_init=L_data,
                                  param_mode="logchol", logdet=True, norm_L=False, **TK),
        # v5: 同上 + trace 归一化 —— 与 fixed/regime 注入等量噪声能量, 只比"形状"
        "hdgn_learned_v5":   dict(hidden=hidden, learn_L=True, L_init=L_data,
                                  param_mode="logchol", logdet=True, norm_L=True, **TK),
        # v7: 去掉 logdet, 改用**外生**的 KL 锚定(见 Arm.kl_to_anchor)。
        # v4/v5/v6 的失败说明补 logdet 只是换了一扇被打开的门 —— C 内生于 Σ 时,
        # 去噪损失对 Σ 无信号, 必须由 Σ_data 这个外生量来定。
        "hdgn_learned_v7":   dict(hidden=hidden, learn_L=True, L_init=L_data,
                                  param_mode="logchol", logdet=False, norm_L=True,
                                  anchor=L_data, **TK),
    }
    # v8: 随 t 变形的协方差 —— 大 t 精确等于 Σ_data(守恒, 低 NFE 的关键),
    # 小 t 才偏离(那里 (1-ᾱ_t)→0, 偏离几乎不影响 Cov(x_t))。
    S["hdgn_learned_v8"] = dict(hidden=hidden, learn_L=True, L_init=L_data,
                                param_mode="tdep", logdet=False, norm_L=False,
                                anchor=L_data, tbins=8, tdep_pow=1.0, **TK)
    # v8b: δ(t)=ᾱ_t² —— 把偏离更集中到小 t, 守恒违反上界从 1/4 降到 4/27
    S["hdgn_learned_v8b"] = dict(hidden=hidden, learn_L=True, L_init=L_data,
                                 param_mode="tdep", logdet=False, norm_L=False,
                                 anchor=L_data, tbins=8, tdep_pow=2.0, **TK)
    # v8c/v8d: δ(t)=ᾱ_t^3 / ᾱ_t^4。实测 p 的效果**非单调**(p=0 的 v7 是 16/25,
    # p=1 的 v8 只有 5/30, p=2 的 v8b 是 20/30) —— 因为守恒违反 ∝ ᾱ_t(1-ᾱ_t),
    # 峰在 ᾱ=0.5 处; p=1 恰好把偏离堆在那个峰上, p 越大越绕开它。
    S["hdgn_learned_v8c"] = dict(hidden=hidden, learn_L=True, L_init=L_data,
                                 param_mode="tdep", logdet=False, norm_L=False,
                                 anchor=L_data, tbins=8, tdep_pow=3.0, **TK)
    S["hdgn_learned_v8d"] = dict(hidden=hidden, learn_L=True, L_init=L_data,
                                 param_mode="tdep", logdet=False, norm_L=False,
                                 anchor=L_data, tbins=8, tdep_pow=4.0, **TK)
    if L_toe is not None:
        # 块 Toeplitz 投影: 强制时间平稳, 自由参数少 8 倍, 去噪的正是滞后结构
        S["hdgn_toeplitz"] = dict(hidden=hidden, learn_L=False, L_init=L_toe, **TK)
    if L_bank is not None:
        # dynamic 臂: 按市场活跃度切 regime, 每个 regime 一个 Sigma_k
        S["hdgn_regime"] = dict(hidden=hidden, learn_L=False, L_init=None,
                                L_bank=L_bank, **TK)
    if eig is not None:
        # v6: 谱族 —— 冻结 Sigma_data 的特征向量, 只学 D 个特征值(参数少 344 倍)
        S["hdgn_learned_v6"] = dict(hidden=hidden, learn_L=True, L_init=None,
                                    param_mode="spectral", logdet=True,
                                    norm_L=True, eig=eig, **TK)
    return S


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
def ddim_sample(model, ab, n, nfe, device, seed=0, Tsteps=1000, x0_clip=None, reg_prior=None):
    """DDIM 采样。

    x0_clip 不是可选的美化项: cosine schedule 令 ᾱ_T -> 0 (实测 ab[999]=2.43e-9),
    而 x0 = (x - sqrt(1-ᾱ)ε_θ)/sqrt(ᾱ) 会把 ε_θ 的误差放大 1/sqrt(ᾱ) = 20,291 倍。
    没有 clip 时, 即使模型把噪声预测得很准(loss 0.15, 解释 85% 方差), x0 也会在
    第一步爆炸, 整条链作废 —— 表现为 loss 很低但生成质量恒定为零。
    这就是 DDPM/DDIM 参考实现里的 clip_denoised。
    """
    g = torch.Generator(device=device); g.manual_seed(seed)
    regimes = None
    if getattr(model, "param_mode", "") == "tdep":
        # DDIM 只在起点抽一次噪声, 起点是 t=T。ᾱ_T=2.43e-9 → δ(T)≈0,
        # 所以先验精确等于 N(0,Σ_data) —— 与 hdgn_fixed 的先验完全一致。
        regimes = torch.full((n,), model.n_tbins - 1, dtype=torch.long, device=device)
    elif getattr(model, "has_bank", False):
        # 采样时按训练集的 regime 频率抽, 保证生成分布的 regime 组成与真实一致
        pr = torch.as_tensor(reg_prior, dtype=torch.float32, device=device)
        regimes = torch.multinomial(pr, n, replacement=True, generator=g)
    x = model.get_noise(n, generator=g, regimes=regimes)
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
    ap.add_argument("--windows-npy", default=None,
                    help="直接加载预生成的 (M,T,C) 窗口张量(如 1 分钟数据), 跳过 tick 级构造")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam1", type=float, default=1e-2)   # 论文 HP_LAMBDA_1 默认
    ap.add_argument("--lam2", type=float, default=0.0)
    ap.add_argument("--toeplitz-blend", type=float, default=1.0,
                    help="hdgn_toeplitz 的投影强度, 1.0=完全平稳")
    ap.add_argument("--lam-kl", type=float, default=1.0,
                    help="v7 的 KL 锚定权重: KL(N(0,Σ_data)‖N(0,Σ))")
    ap.add_argument("--lam-anchor", type=float, default=1.0,
                    help="v3: 惩罚 L 偏离 Sigma_data 的幅度")    # 论文 HP_LAMBDA_2 默认
    ap.add_argument("--n-gen", type=int, default=4000)
    ap.add_argument("--eval-nfe", type=int, default=50)
    ap.add_argument("--nfe-sweep", default="5,10,20,50,100,200")
    ap.add_argument("--arms", default="iid,iid_wide,hdgn_fixed,hdgn_learned")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conv-window", type=int, default=20,
                    help="收敛判据: 用最近 N 段(每段 500 步)的 loss 均值比较")
    ap.add_argument("--conv-eps", type=float, default=0.002,
                    help="收敛判据: 相邻两窗 loss 均值的相对变化阈值")
    ap.add_argument("--min-steps", type=int, default=20000,
                    help="收敛判据生效前的最小步数")
    ap.add_argument("--n-regimes", type=int, default=3,
                    help="dynamic 臂的市场 regime 数, 按窗口活跃度分位数切分")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = lambda *a: print(*a, flush=True)

    dev = args.device
    if args.windows_npy:
        X = np.load(args.windows_npy).astype(np.float64)
        rng = np.random.default_rng(args.seed); rng.shuffle(X)
        args.T = X.shape[1]
        log(f"[setup] 从 {args.windows_npy} 加载窗口 {X.shape}")
    else:
        X = load_dataset(DS, T8, T=args.T, stride=args.T,
                         max_rows_per_ticker=args.rows, seed=args.seed)
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
    # 谱族(v6)用的特征分解。Sigma 对称正定, 用 eigvalsh 族而非 SVD
    # (688 维上实测 SVD 会不收敛, 见 _cond 处的同一条教训)。
    eig_lam, eig_Q = np.linalg.eigh(Sigma)
    eig_lam = np.clip(eig_lam, 1e-12, None).astype(np.float32)
    eig_Q = eig_Q.astype(np.float32)
    eff_rank_data = float((eig_lam.sum() ** 2) / (eig_lam ** 2).sum())
    # 块 Toeplitz 投影(hdgn_toeplitz 臂): 强制时间平稳, 自由参数 T²C²/2 -> T·C²。
    # 留出集实测同时改善似然(3.1563->3.1581)与跨时间块误差(0.5544->0.5520),
    # 是唯一一个"改进方向"与"评估指标(ACF/签名)"对齐的候选, 见 REPORT 13H.5。
    L_toe = None
    try:
        S_toe = noise_covariance_toeplitz(Sigma, args.T, 43, blend=args.toeplitz_blend)
        L_toe = np.linalg.cholesky(S_toe).astype(np.float32)
    except Exception as e:
        log(f"[setup] Toeplitz 投影不可用({e}), 跳过 hdgn_toeplitz 臂")

    # ---- dynamic 臂: 按市场活跃度切 regime, 每个 regime 估一个 Sigma_k ----
    # col0 是活跃度代理(tick 级 = log_dt 越小越活跃; horizon 级 = log 事件数 越大越活跃)。
    # 用它是因为它正是波动率聚集所在的那个量(1 horizon 数据上 lag-1 自相关 0.954)。
    # 价格已转成相对 mid 的 tick 偏移, 特征空间里没有绝对价, 因此不能用已实现波动率。
    K = args.n_regimes
    act = Xz[:, :, 0].mean(1)
    cuts = np.quantile(act[:ntr], np.linspace(0, 1, K + 1)[1:-1])
    regimes_all = np.digitize(act, cuts).astype(np.int64)
    L_bank, reg_counts = [], []
    for k in range(K):
        m = (regimes_all[:ntr] == k)
        reg_counts.append(int(m.sum()))
        S_k = noise_covariance(Xtr[m])          # 同样归一化到 trace=D, 保证与 fixed 臂等能量
        L_bank.append(np.linalg.cholesky(S_k).astype(np.float32))
    L_bank = np.stack(L_bank)
    reg_prior = np.array(reg_counts, dtype=np.float64) / max(sum(reg_counts), 1)
    # Sigma 是对称正定的, 用 eigvalsh 而非走 SVD 的 np.linalg.cond(688 维上实测 SVD 会
    # 不收敛)。并且这只是诊断打印, 绝不能有能力杀死训练 —— 故整体 try 保护。
    def _cond(M):
        try:
            w = np.clip(np.linalg.eigvalsh(M), 1e-30, None)
            return float(w.max() / w.min())
        except Exception:
            return float("nan")
    conds = [_cond(L_bank[k] @ L_bank[k].T) for k in range(K)]
    log(f"[setup] regime K={K} 训练样本数={reg_counts} 先验={np.round(reg_prior,3).tolist()}")
    log(f"[setup] 各 regime 的 Σ 条件数={[f'{c:.2e}' for c in conds]} (单一 Σ: {cond:.2e})")
    real_F = normer.inverse_transform(Xte.reshape(-1, args.T, 43))
    # x0 clip 阈值: 取训练数据的 99.99% 分位数再留 20% 余量, 保证不裁掉真实数据
    x0_clip = float(np.quantile(np.abs(Xtr), 0.9999) * 1.2)
    log(f"[setup] dev={dev} 窗口{X.shape} D={D} train={len(Xtr)} test={len(Xte)} Σcond={cond:.3e}")
    log(f"[setup] x0_clip={x0_clip:.2f} (训练数据 |z| 的 max={np.abs(Xtr).max():.2f})")
    log(f"[setup] Σ_data 有效秩={eff_rank_data:.2f}/{D} "
        f"top10 特征值占能量 {eig_lam[::-1][:10].sum()/eig_lam.sum()*100:.1f}% "
        f"—— 可学臂若把它推向 {D} 就是学成了 iid(见 13H.1)")

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
               "n_train": len(Xtr), "n_test": len(Xte), "hidden_wide": Hw, "x0_clip": x0_clip, "norm": args.norm, "n_regimes": K,
               "regime_counts": reg_counts, "regime_sigma_cond": conds,
               "features_covered": list(COVERED), "features_not_covered": list(NOT_COVERED),
               "baselines": {"floor_real_vs_real": floor, "gaussian_sigma_untrained": gauss},
               "arms": {}}

    SPEC = arm_specs(args.hidden, Hw, L_data, L_bank=L_bank,
                     eig=(eig_lam, eig_Q), L_toe=L_toe,
                     trunk=args.trunk, depth=args.depth)

    Xtr_t = torch.as_tensor(Xtr, device=dev)
    regimes_t = torch.as_tensor(regimes_all[:ntr], device=dev)
    for arm in args.arms.split(","):
        torch.manual_seed(args.seed)                        # 四臂同初始化种子
        model = Arm(args.T, 43, **SPEC[arm]).to(dev)
        npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        gen = torch.Generator(device=dev); gen.manual_seed(args.seed + 1)   # 同数据顺序
        # **约束来源闸门**: 可学的 Σ 必须至少有一个东西拦住 L→∞ 的平凡解。
        # 没有它, v8 那次的失败方式是: 正则静默不生效 -> 完全无约束 -> trace 涨 180 倍,
        # 而 loss 反而更好看(0.0378), 全程不报错。改成开跑前大声说出来。
        _spec = SPEC[arm]
        _src = ([n for n, on in (("logdet", _spec.get("logdet")),
                                 ("trace归一化", _spec.get("norm_L")),
                                 ("KL/Frobenius锚", _spec.get("anchor") is not None),
                                 ("lam1 正则", args.lam1 > 0)) if on])
        if _spec.get("learn_L"):
            if not _src:
                log(f"[train] !! arm={arm} 的可学 Σ **没有任何约束** —— "
                    f"这会跌进 L→∞ 的平凡解(见 13H.2/13H.9)。若是故意的演示请忽略。")
            else:
                log(f"[train] arm={arm} Σ 约束来源: {', '.join(_src)}")
        log(f"[train] arm={arm} 参数={npar:,}")
        hist, evs = [], {}
        converged_at = None
        t0 = time.time()
        for step in range(1, args.steps + 1):
            idx = torch.randint(0, len(Xtr_t), (args.batch,), device=dev, generator=gen)
            x0 = Xtr_t[idx]
            t = torch.randint(0, 1000, (args.batch,), device=dev, generator=gen)
            # tdep 臂的分组索引是 **t 分箱**(每步不同), regime 臂才是市场态(每样本固定)。
            reg_b = (model.tbin_of(t) if model.param_mode == "tdep"
                     else (regimes_t[idx] if model.has_bank else None))
            eps = model.get_noise(args.batch, generator=gen, regimes=reg_b)
            a = ab[t][:, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            r = model(xt, t) - eps
            loss = model.mahalanobis(r, reg_b)
            # 白名单必须覆盖**所有**用 KL 锚的参数化。血泪教训(2026-08-08):
            # 这里原本只写了 "logchol", 新增的 "tdep" 不在名单里 —— 于是 v8/v8b 的
            # KL 锚**从未被加上**, 而它们同时 logdet=False、norm_L=False,
            # 结果是完全无约束: 学到的 Σ trace 涨到 123700(180 倍)、有效秩 302,
            # 直接跌进 13H.2 推导的 L→∞ 平凡解。**不报错, 且 loss 反而更好看**(0.0378)。
            # 改为按"有锚点就用 KL"判定, 不再枚举 param_mode。
            if (model.learn_L and not model.use_logdet
                    and model.L_anchor.numel() > 0 and args.lam_kl > 0):
                loss = loss + args.lam_kl * model.kl_to_anchor()
            if model.learn_L and model.use_logdet:
                # 高斯 NLL 的归一化常数。加上它之后不需要 lam1(平凡解自动被堵死),
                # 故 v4/v5/v6 都跑在 lam1=0 上 —— 这不是省事, 是让目标回到良定义。
                loss = loss + model.logdet_over_D()
            L = model.get_L() if (model.learn_L and model.L_params is not None) else None
            if model.learn_L and L is not None:
                loss = loss + args.lam1 * (L ** 2).sum() + args.lam2 * (torch.diag(L) ** 2).sum()
                if model.L_anchor.numel() > 0 and args.lam_anchor > 0:
                    # 惩罚偏离 Sigma_data 的幅度, 使学习是"微调"而非"重学"
                    loss = loss + args.lam_anchor * (model.L_params ** 2).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 100 == 0:
                hist.append((step, loss.detach().item()))
                # 收敛判据: 最近 conv_window 段与前一组同长窗口的 loss 均值相对变化 < eps。
                # 用固定步数会让"训练到收敛"变成假定而非事实(不同臂收敛速度差很多)。
                W_ = args.conv_window
                if step >= args.min_steps and len(hist) >= 2 * W_:
                    a = np.mean([v for _, v in hist[-2 * W_:-W_]])
                    b = np.mean([v for _, v in hist[-W_:]])
                    # 分母下界取 1.0: v4 补了 logdet 后 loss 会是负的(≈ 1 - 3 = -2),
                    # 训练途中可能跨零, 那时 |a-b|/|a| 会失去意义 —— 是一个不报错的静默故障。
                    # 所有臂的 loss 都是 O(1), 故这等价于"接近零时退化成绝对阈值",
                    # 且对已有臂只会更严格(更接近真收敛), 方向与任务 13/16 一致。
                    den = max(abs(a), abs(b), 1.0)
                    if abs(a - b) / den < args.conv_eps:
                        converged_at = step
                        log(f"    [{arm}] 收敛于 step {step} "
                            f"(loss {a:.4f} -> {b:.4f}, 相对变化 {abs(a-b)/den:.2e})")
                        break
            if step in eval_at:
                model.eval()
                g = ddim_sample(model, ab, args.n_gen, args.eval_nfe, dev, seed=1234, x0_clip=x0_clip, reg_prior=reg_prior)
                gen_F = normer.inverse_transform(g.float().cpu().numpy().reshape(-1, args.T, 43))
                per, means = score_all(real_F, gen_F)
                evs[step] = {**means, "_per": per}
                model.train()
                log(f"    [{arm}] step {step:6d} loss {loss.item():.4f} "
                    f"l1={means['l1']:.4f} ws={means['wasserstein']:.4f} ks={means['ks']:.4f}")
        wall = time.time() - t0
        # **先存权重, 再做任何分析。** 血泪教训(2026-08-08): 下面这段诊断在 tdep 臂上
        # 抛 TypeError, 而它排在 torch.save 之前 —— 一个已经收敛的臂(v8b @ step 51000)
        # 因此整个丢掉。产物落盘不能依赖任何分析代码的正确性。
        # 这是"诊断不该有能力杀死训练"那条教训的结构版: 上次只给出错的那一行包了
        # try/except, 没有改顺序, 于是同一个病换个地方又犯了一次。
        torch.save(model.state_dict(), os.path.join(args.out, f"model_{arm}.pt"))
        # 学到的 Sigma 塌了没有: 有效秩往 D 跑 = 学成了 iid(13H.1 的判据)。
        # 这是每个可学臂训练后立刻要看的一个数, 比 loss 更能说明它到底学到了什么。
        learned_er = None
        if model.learn_L:
            try:
                with torch.no_grad():
                    Lh = model.get_L().float().cpu().numpy()
                    w = np.clip(np.linalg.eigvalsh(Lh @ Lh.T), 0, None)
                    learned_er = float((w.sum() ** 2) / max((w ** 2).sum(), 1e-30))
                log(f"    [{arm}] 学到的 Σ 有效秩 {learned_er:.2f} "
                    f"(Σ_data={eff_rank_data:.2f}, 各向同性={D}) "
                    f"trace={float(np.trace(Lh @ Lh.T)):.1f} (Σ_data={D})")
            except Exception as e:
                log(f"    [{arm}] 有效秩诊断失败({e}), 不影响训练产物")
        nfe_res = {}
        model.eval()
        for nfe in [int(x) for x in args.nfe_sweep.split(",")]:
            ts = time.time()
            g = ddim_sample(model, ab, args.n_gen, nfe, dev, seed=1234, x0_clip=x0_clip, reg_prior=reg_prior)
            gen_F = normer.inverse_transform(g.float().cpu().numpy().reshape(-1, args.T, 43))
            _, means = score_all(real_F, gen_F)
            nfe_res[str(nfe)] = {**means, "sample_s": time.time() - ts}
            log(f"    [{arm}] NFE={nfe:4d} l1={means['l1']:.4f} ws={means['wasserstein']:.4f} ks={means['ks']:.4f}")
        results["arms"][arm] = {"n_params": npar, "wall_s": wall, "loss_hist": hist,
                                "converged_at": converged_at, "stopped_at": step,
                                "learned_eff_rank": learned_er, "eff_rank_data": eff_rank_data,
                                "evals": {str(k): v for k, v in evs.items()}, "nfe_sweep": nfe_res}
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log(f"[train] arm={arm} 完成, 用时 {wall:.1f}s")
    log(f"[done] -> {os.path.join(args.out,'results.json')}")


if __name__ == "__main__":
    main()
