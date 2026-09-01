"""复现 P0/E7 的挂起点:导入 slime 栈之后再 ray.init。

现象(2026-09-01,两次独立运行同样卡在这里):train.py 打印
"Creating placement group with N GPUs..." 与 "Connected to Ray cluster" 之后
无限期停住,CPU 0%、GPU 全空、Ray 集群健康且无 pending demand。

py-spy 给出位置:
    connect (ray/_private/worker.py:2722)   <- ray._raylet.CoreWorker(...) 构造
    init (ray/_private/worker.py:2026)
    auto_init_ray (ray/_private/auto_init_hook.py:15)
    _create_placement_group (slime/ray/placement_group.py:48)
主线程的内核等待点是 unix_stream_read_generic —— 在等 raylet 经 Unix socket 的回复。
进程有 156 个线程(torch/OpenMP 在 288 核上开的)。

已排除:环境变量(LD_LIBRARY_PATH/CUDA_VISIBLE_DEVICES/CUDA_DEVICE_MAX_CONNECTIONS)、
先建 CUDA 上下文 —— 这两种情况下 ray.init 都是 0.4 秒。
本探针检验最后一个差别:**先导入整个 slime 栈**(sglang + megatron)再 ray.init。
"""
import os
import sys
import time

t0 = time.time()
from slime.utils.arguments import parse_args  # noqa: F401  拉进 sglang + megatron

nthreads = len(os.listdir("/proc/self/task"))
print(f"slime 导入完成 {time.time() - t0:.0f}s,线程数 {nthreads}", flush=True)

import ray  # noqa: E402

t = time.time()
ray.init(address="auto", log_to_driver=False)
dt = time.time() - t
print(f"ray.init 成功,用时 {dt:.1f}s", flush=True)
print("资源:", ray.cluster_resources(), flush=True)
sys.exit(0)
