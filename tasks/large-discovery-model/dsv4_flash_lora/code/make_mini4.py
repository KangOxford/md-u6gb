#!/usr/bin/env python
"""从本地全量权重造 4 层 mini 版 DeepSeek-V4-Flash。

与官方文档 mini 流程的差异:不做任何反量化——原样保留 FP4/FP8 权重、scale 与
quantization_config,让 mcore-bridge 在加载时走与全量模型完全相同的转换路径。
(文档反量化是为了 FP32 精度对齐测试;我们的用途是全链路训练冒烟,保真更重要。
另:Fp8Dequantizer 不认 FP4 融合专家张量的打包布局,2026-08-26 实测 reshape 崩。)
"""
import json
import os
import re
import shutil
import sys

import torch
from mcore_bridge.utils import SafetensorLazyLoader, StreamingSafetensorSaver

SRC = "/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash"
DST = "/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash-mini4"
KEEP = 4

if os.path.exists(os.path.join(DST, "model.safetensors.index.json")):
    print(f"[mini4] {DST} 已存在,跳过构建")
    sys.exit(0)

os.makedirs(DST, exist_ok=True)
loader = SafetensorLazyLoader(SRC)
sd = loader.get_state_dict()

layer_re = re.compile(r"(?:model\.)?layers\.(\d+)\.")
max_idx = max((int(m.group(1)) for k in sd if (m := layer_re.match(k))), default=-1)
print(f"[mini4] 张量总数 {len(sd)}  最大层号 {max_idx}", flush=True)

saver = StreamingSafetensorSaver(save_dir=DST)
kept = dropped = 0
for k, v in sd.items():
    m = layer_re.match(k)
    if m and int(m.group(1)) >= KEEP:
        dropped += 1
        continue
    saver.add_tensor(k, v if isinstance(v, torch.Tensor) else v.load())
    kept += 1
saver.finalize()
print(f"[mini4] 保留 {kept}  丢弃 {dropped}(量化原样,不反量化)", flush=True)

cfg = json.load(open(f"{SRC}/config.json"))
cfg["num_hidden_layers"] = KEEP
cfg["compress_ratios"] = [0, 0, 4, 128, 0]   # 官方文档给的 4 层 mini 取值
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if os.path.exists(f"{SRC}/{f}"):
        shutil.copy(f"{SRC}/{f}", DST)
print(f"[mini4] DONE -> {DST}", flush=True)
