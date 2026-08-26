#!/usr/bin/env python
"""下载 DeepSeek-V4-Flash 权重(可断点续传,重跑安全)。

登录节点 nproc 上限 1900:hf_xet 的多线程 rust 运行时会把线程配额打穿导致
SIGKILL(2026-08-26 实测 25/73 处无声死亡),故强制关闭 xet 走普通 HTTP。
"""
import faulthandler
import os
import sys
import time
import traceback

faulthandler.enable()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"          # 关键:禁 xet,防线程爆炸
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from huggingface_hub import snapshot_download

REPO = "deepseek-ai/DeepSeek-V4-Flash"
DEST = "/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash"

t0 = time.time()
for attempt in range(1, 11):
    try:
        print(f"[download] 第 {attempt} 次尝试  {REPO} -> {DEST}", flush=True)
        path = snapshot_download(repo_id=REPO, local_dir=DEST, max_workers=8)
        print(f"[download] DONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        print(f"[download] 第 {attempt} 次失败,30s 后重试", flush=True)
        time.sleep(30)
print("[download] FAILED after 10 attempts", flush=True)
sys.exit(1)
