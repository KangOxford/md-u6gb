#!/usr/bin/env python
"""从本地全量权重造 4 层 mini 版 DeepSeek-V4-Flash(照官方最佳实践的做法,改为全本地)。

用途:几分钟内验证 mcore-bridge 加载、DSA/CSA 内核、LoRA 全链路,
不用等全模型 500GB 的转换。在计算节点上跑(单个专家融合张量解包后 >12GB RAM)。
"""
import json
import os
import re
import shutil
import sys

import torch
from mcore_bridge.utils import Fp8Dequantizer, SafetensorLazyLoader, StreamingSafetensorSaver

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
suffixes = sorted({k.rsplit(".", 1)[-1] for k in sd})
print(f"[mini4] 张量总数 {len(sd)}  最大层号 {max_idx}  后缀集合 {suffixes}", flush=True)

SCALE_SUFFIXES = (".scale", ".weight_scale_inv", ".weight_scale")


def find_scale(key):
    if key.endswith(".weight"):
        stem = key[: -len(".weight")]
        for s in (stem + ".scale", stem + ".weight_scale_inv", stem + ".weight_scale", key + "_scale_inv"):
            if s in sd:
                return s
    return None


deq = Fp8Dequantizer()
saver = StreamingSafetensorSaver(save_dir=DST)
kept = dropped_layers = dequanted = 0
for k, v in sd.items():
    m = layer_re.match(k)
    if m and int(m.group(1)) >= KEEP:
        dropped_layers += 1
        continue
    if k.endswith(SCALE_SUFFIXES):
        continue
    sk = find_scale(k)
    if sk is not None:
        v = deq.convert(v.load(), sd[sk].load()).to(torch.bfloat16)
        dequanted += 1
    saver.add_tensor(k, v if isinstance(v, torch.Tensor) else v.load())
    kept += 1
saver.finalize()
print(f"[mini4] 保留 {kept}(其中反量化 {dequanted})  丢弃层张量 {dropped_layers}", flush=True)

cfg = json.load(open(f"{SRC}/config.json"))
cfg["num_hidden_layers"] = KEEP
cfg["compress_ratios"] = [0, 0, 4, 128, 0]   # 官方文档给的 4 层 mini 取值
cfg.pop("quantization_config", None)
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if os.path.exists(f"{SRC}/{f}"):
        shutil.copy(f"{SRC}/{f}", DST)
print(f"[mini4] DONE -> {DST}", flush=True)
