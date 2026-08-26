#!/usr/bin/env python
"""预取三个训练数据集到共享缓存,免得 16 节点各自下载互相踩。

用 ms-swift 自己的加载器,保证缓存布局与训练时一致。
"""
import os

os.environ.setdefault("MODELSCOPE_CACHE", "/lus/lfs1aip2/projects/public/u6gb/models/ms_cache")
os.environ.setdefault("HF_HOME", "/lus/lfs1aip2/projects/public/u6gb/models/hf_home")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from swift.dataset import load_dataset   # ms-swift 4.6 重构:swift.llm 已拆散

datasets = [
    "AI-ModelScope/alpaca-gpt4-data-zh#1000",
    "AI-ModelScope/alpaca-gpt4-data-en#1000",
    "swift/self-cognition#1000",
]
train, val = load_dataset(
    datasets,
    split_dataset_ratio=0.01,
    seed=42,
    num_proc=4,
    model_name=["swift-robot"],
    model_author=["swift"],
)
print("train rows:", len(train), " val rows:", len(val))
print("样例:", train[0])
print("PREFETCH_DONE")
