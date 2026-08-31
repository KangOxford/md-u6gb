# LDM-RL(小分子 acquisition-RL)在 Isambard-AI 上的落地计划

**来源**:陈奕杭(汪老师组)交接。代码 `YihangChen9/Large-Discovery-Models` @ `rl` 分支
(本地 HEAD `5ca5efe9`),交接文档 `rl/slime_launch/HANDOFF.md` + `RUNS.md` + `TRAINING_PLAN.md`。

**要做的事**:GRPO(on Slime)在 SFT 之上继续强化「acquisition 倾向的搜索策略」。
**训 KRAS G12D → 评 G12C(迁移)+ G12D(同分布)**,指标是 Pareto 前沿 hypervolume。

| Run | 起点 | reward | 回答什么 |
|---|---|---|---|
| R1 | base Qwen3.5-9B | acquisition-max | 与 R2 比 → RL 需不需要 SFT 打底 |
| R2 | SFT(no-GP) | acquisition-max | 枢轴。与「SFT 不训」比 → RL 有没有增益 |
| R3 | SFT | hypervolume(真实结果) | 与 R2/R4 比 → 哪种 reward 信号最好 |
| R4 | SFT | acquisition-mean | 同上 |

---

## 1. 本机与作者机器的差别(决定了要改什么)

| | 作者机器 | Isambard-AI |
|---|---|---|
| 架构 | x86_64 | **aarch64 (GH200 Grace-Hopper)** |
| 运行方式 | 单机 docker,root | **Slurm 多用户共享,无 root,Lustre** |
| 路径 | `/mnt/data0/ys/LDM`、`/root/megatron-lm` | 全部要参数化 |
| 环境 | micromamba `slime` in container | conda env on `/home`(VAST,不吃 Lustre inode) |
| CUDA | 12.9 | 驱动 565 = **CUDA 12.7**,同 major 前向兼容 cu128/cu129 |

**Isambard 侧的额外硬约束**(与作者机器无关,但违反会被封号):
- Lustre inode 已用 **97.4%**(49.85M / 51.2M)。**环境一律建在 `/home`(VAST 文件系统)**,
  只有源码和权重放 Lustre。
- 登录节点禁止编译、禁止 GPU。**所有源码编译步骤必须在计算节点上跑。**
- 每次 `sbatch` 前先 `gtop`;名下有空卡就 attach,不排队。

---

## 2. aarch64 逐项落地表(本计划的核心)

交接文档 §7 把 **TE / sglang 在 ARM 上装不上**列为头号风险。逐项查证后,
**大部分件现在都有 aarch64 预编译轮子**,真正要源码编的只剩三个。

| build_conda.sh 里的件 | 钉的版本 | aarch64 现状 | 本机做法 |
|---|---|---|---|
| python | 3.12 | — | 同(小分子栈要求 `>=3.11,<3.13`) |
| torch / torchaudio | `2.11.0+cu129` | ✅ `manylinux_2_28_aarch64` | **原版照用** |
| sglang | `v0.5.15.post1` | ✅ cp310–cp313 aarch64 轮子 | **用 PyPI 轮子**,不走源码 `-e`(源码要 Rust 编 sglang-grpc) |
| sglang-kernel | `0.4.4+cu129` | ✅ `manylinux2014_aarch64` | **原版照用** |
| sgl-deep-gemm | `0.1.4+cu129` | ⚠️ 索引里最低 `0.1.5rc3` | 用 `0.1.5rc3+cu129` aarch64 |
| transformer_engine | `2.16.1` | ❌ 2.16.1 **只有 x86** | **用 `transformer_engine_cu12==2.16.0`**(有 aarch64,差一个补丁号) |
| flash-attn | `2.8.3` 社区轮子 | ❌ 该轮子是 `linux_x86_64` | **源码编**(GH200 有前例,~10–40 min) |
| flash-linear-attention | `0.4.2` | ✅ `py3-none-any` | 原版照用 |
| FlashQLA + tilelang | — | ❌ tilelang nightly **只有 x86** | **跳过**。FlashQLA 是 Qwen3.5 GDN 的*可选*后端,默认 FLA(triton)后端可用;slime 自己的 Dockerfile 在 CUDA13 分支也是这么绕的 |
| apex | git commit | 源码编 | **跳过**。启动脚本已带 `--no-gradient-accumulation-fusion` |
| torch_memory_saver | `8d30c59` | 源码(需 nvcc) | **源码编**。必须 `--no-build-isolation`,否则出的是 46KB 纯 python 轮子,sglang 会报 `Only hook_mode=preload supports pauseable CUDA Graph` |
| sglang_router(slime fork) | `v0.3.2-9daabcd` | ❌ release 只有 x86 | **源码编**(Rust,conda-forge 装 rust) |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | 源码 `-e`,编 `helpers_cpp` | 同,必须 `--no-build-isolation` |
| int4_qat kernel | slime 内 | CUDA 编 | 编;失败则看是否为必需 |
| CUDA toolkit | conda `cuda=12.9.1` | conda-forge 有 aarch64 | 同(TE/flash-attn/TMS 编译要 `CUDA_HOME`) |

**结论:要源码编的只有 flash-attn、torch_memory_saver、sglang_router、Megatron 的
helpers_cpp、int4_qat 五处**,全部是 CPU 编译,GH200 每节点 288 核,可高并发。
**TE 这块最大的风险被预编译轮子消掉了。**

### 2.1 cu13 溢出问题
sglang 0.5.15.post1 的元数据要 `flashinfer_python[cu13]` + PyPI 默认 cu13 的 torch,
而驱动 565 只到 CUDA 12.7 —— 装上去 `torch.cuda.is_available()` 直接 False(交接文档 §1)。
作者的解法是**装完再强制回退**:`--force-reinstall --no-deps` 把 torch / sglang-kernel /
sgl-deep-gemm 换成 `+cu129`,再把 `nvidia-*-cu13` 卸掉换 `-cu12`。本机照做。
作者还用 `touch /root/cudart_block/libcudart.so.13` 加 `LD_LIBRARY_PATH` 首位来堵住
系统里的 cudart 13 —— **本机没有系统级 CUDA 13**,这一步先不做,若 TE 的 cudnn-frontend
检查报 cu13 再补。

---

## 3. 两个环境,不是一个

`rl/ldm_rl/remote_env.py` 里 `RemoteLDMEnv` 用 **`task_python` 起子进程**,走 JSON-lines
stdio,并显式剥掉 `LD_LIBRARY_PATH` / `CUDA_HOME` / `CUDA_VISIBLE_DEVICES`。
即:**评测栈本来就跑在另一个 Python 里**,这是代码写死的边界,不是我的选择。

| 环境 | 路径 | 装什么 | 风险 |
|---|---|---|---|
| **评测栈** | `/home/u6gb/kangli.u6gb/envs/ldm-rl` | rdkit / gpytorch / gauche / lightgbm / meeko / gemmi / vina + torch(CPU 用) | 低。全是现成 aarch64 轮子 |
| **训练栈** | `/home/u6gb/kangli.u6gb/envs/ldm-rl-train` | torch2.11+cu129 / TE / sglang / Megatron / slime / ray | 高。见上表 |

分开的另一个理由:训练栈的装法里有大量 `--force-reinstall --no-deps` 和
`pip uninstall nvidia-*`,是个脆弱环境;把定义 reward 的评测栈绑在它上面,
重建一次就连累另一边。

---

## 4. 执行顺序(先做不依赖未解问题的部分)

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **E0** | 评测栈环境 + vina(aarch64) | 无 | 进行中 |
| **E1** | `ldm_rl` 单元测试(53 个)跑通 | E0 | |
| **E2** | `small_molecule_real_smoke.py` —— **真 vina + 真活性模型 + 真 reward**,纯 CPU | E0 + 8UN5 受体 | |
| **E3** | 训练栈环境(§2 那张表) | 计算节点(要编译) | |
| **E4** | **P0 关卡**:Qwen2.5-1.5B GRPO 冒烟,验 backward 不 SIGSEGV | E2+E3 | |
| **E5** | 模型下载 + HF→Megatron torch_dist 转换(base + SFT) | E3 | |
| **E6** | **P1**:9B real 极小 count 冒烟(count=1, iterations=2),验 hybrid backward / 显存 | E4+E5 | |
| **E7** | 暖共享 GP(`run_warmup_real_slime.sh`,dock 60 个分子) | E2 | |
| **E8** | **P2**:R1–R4 × 3–5 seed 真训练 | 全部 | |
| **E9** | 离线评测 G12C + G12D,填对照表;并行训 `best_g12c_model.joblib` | E8 | |

**E2 是第一个真结果**:它不需要 GPU、不需要 slime,却验证了整条 reward 路径
(docking → 活性 → GP/SIR → acquisition/hypervolume)。这条路径是四个 run 共用的,
坏了什么都别谈。

---

## 5. 已知会踩的坑(记在前面)

1. **启动脚本全是写死路径**(`/mnt/data0/ys/LDM`、`/root/megatron-lm`、
   `/root/micromamba/envs/slime`)。必须参数化后才能在 Slurm 上跑。
2. **`ray start --head --node-ip-address 127.0.0.1`** 在多节点上不成立;单节点 4 卡先按
   127.0.0.1 跑通,再考虑跨节点。
3. **`config_real.json` 里三条路径要改**:`gp_history_file`、`vina_bin`、
   `nn_model_path`、`output_dir`。
4. **`vina_max_workers=32`**:vina 是纯 CPU,GH200 每节点 288 核,可继续加大;
   但 docking 是真瓶颈,**先量一个分子的真实耗时再定并发**。
5. **不要自设 `CUDA_VISIBLE_DEVICES`**——`srun --gres=gpu:N` 已经隔离好卡。
   但注意 `run_train_real_9b.sh` 里 actor 与 rollout 是按 `CUDA_VISIBLE_DEVICES=0,1,2,3`
   在**同一节点内**分卡的(actor 2 卡 + sglang 2 卡),这个是 slime 自己的分配,要保留。
6. **`--gres=gpu:4` 的作业要落在 4/4 全空的节点上**;部分空会在启动窗口之后 SIGABRT。
