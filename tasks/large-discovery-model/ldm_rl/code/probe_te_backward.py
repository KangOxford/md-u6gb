"""最小的 TE backward 探针:在 GH200/aarch64 上跑一次前向+反向,看会不会 SIGSEGV。

HANDOFF §6/§7 把「aarch64 上 TE 是 ABI 最敏感的一环,backward 可能 SIGSEGV」
列为头号风险,并要求用 1.5B 冒烟来验。但那要先转检查点、起 ray、起 sglang,
一次十几分钟;而且真挂掉时,症状混在一堆分布式错误里。

这里只做能独立回答那个问题的最小实验:构造一个 TE 的 Linear + LayerNormMLP,
bf16 前向、反向、看梯度是不是有限值。几秒钟,且失败时的栈就是 TE 自己的。

装得上 != ABI 对得上。这个探针是两者之间的那一步。
"""
from __future__ import annotations
import sys

import torch

print(f"torch {torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL: torch 看不见卡,后面没意义"); sys.exit(2)
print(f"device: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")

import transformer_engine.pytorch as te
import transformer_engine

print(f"transformer_engine {getattr(transformer_engine, '__version__', '?')}")

torch.manual_seed(0)
dev = "cuda"
B, S, H = 2, 128, 512

# 两个最常用的 TE 模块:普通 Linear 与融合了 LayerNorm 的 MLP。
# GRPO 的 backward 走的就是这一类算子。
net = torch.nn.Sequential(
    te.Linear(H, H, bias=True),
    te.LayerNormMLP(H, 4 * H),
).to(dev)

x = torch.randn(B * S, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y = net(x)
    loss = y.float().pow(2).mean()

print(f"前向 OK  out={tuple(y.shape)}  loss={loss.item():.6f}")
loss.backward()
torch.cuda.synchronize()

gsum = sum(p.grad.float().abs().sum().item() for p in net.parameters() if p.grad is not None)
gcnt = sum(1 for p in net.parameters() if p.grad is not None)
xg = x.grad.float().abs().sum().item()
print(f"反向 OK  有梯度的参数张量 {gcnt} 个  |grad| 求和 {gsum:.4f}  输入梯度 {xg:.4f}")

bad = [n for n, p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
if bad or not torch.isfinite(x.grad).all():
    print(f"FATAL: 出现非有限梯度: {bad}"); sys.exit(1)
if gsum == 0.0:
    print("FATAL: 所有梯度都是 0,反向没有真的算"); sys.exit(1)

print()
print("TE_BACKWARD_OK  aarch64 上 TE 的前向与反向都跑通,梯度有限且非零。")
