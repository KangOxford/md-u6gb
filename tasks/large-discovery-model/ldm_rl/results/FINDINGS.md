# LDM-RL 在 Isambard-AI 上的落地：实测结果

记录**测出来的数**，不记录计划。计划在 `../PLAN.md`。
起于 2026-09-01。分支 `isambard-port`（基于 `rl` 分支 `5ca5efe9`）。

---

## 1. 已完成的关卡

| 阶段 | 内容 | 结果 |
|---|---|---|
| **E0** | 评测栈环境 | ✅ rdkit 2025.09.6 / gpytorch 1.15.2 / gauche 0.1.6 / lightgbm 4.7.0 / meeko 0.8.0 / gemmi 0.7.5 全部 import 通过 |
| | vina 二进制 | ✅ `AutoDock Vina f458505-mod`（conda-forge aarch64），装进 `envs/ldm-rl/bin/vina` |
| **E1** | `ldm_rl` 单元测试 | ✅ **61 passed in 1.37s**（交接文档说 53，rl 分支后续又加了 8） |
| **E2** | 真 reward 全链路 | ✅ 见下表 |
| — | 四组 episodes 数据 | ✅ 60 + 16×3 行，站点路径已烤入，零残留作者路径 |
| — | 三个模型下载 | ✅ 1.5B 2.9G / Qwen3.5-9B 19G / LDM-CoT-SFT 18G |
| — | 启动脚本参数化 | ✅ 10 个脚本，`/mnt/data0` 与 `/root` 清零 |

### E2：真 vina + 真活性模型 + 真 reward（纯 CPU，2 轮，确定性提议器）

三种 reward 跑在**完全相同的分子**上（vina 与 activity 逐轮一致），所以差别只来自 reward 本身：

| reward | round 0 | round 1 | 合计 |
|---|---:|---:|---:|
| `hypervolume` | 0.000000 | 0.000001 | 1e-06 |
| `acquisition` | 0.000000 | 0.000000 | 0.0 |
| `improvement` | 9.365461 | 0.503000 | 9.868461 |

共同的测量值：`vina = -3.395 / -3.898 kcal·mol⁻¹`，`activity(pIC50) = 5.9705 / 5.9003`。

---

## 2. acquisition reward 的 0.0 是真值，不是静默回退

**为什么要查**：`env.py:647-651` 在 `scores` 为空时 `return 0.0`，与「真的算出 0」在数值上不可分。
若是前者，reward 恒为零 ⇒ GRPO 优势全零 ⇒ **没有梯度信号**，而 acquisition 正是 R1/R2 的 reward，
R2 是整个设计的枢轴。

**判定依据落在结构上而不是数值上**：看 `components['scores']` 是不是空列表。

```
round=0 reward=0.000000 scores=[0.0]                  agg=max
round=1 reward=0.000000 scores=[0.0]                  agg=max
round=2 reward=0.063888 scores=[0.06388776076585678]  agg=max
```

**结论：非空 ⇒ 真值。** 前两轮为 0 是 **GP 冷启动**——`reservoir_size=2` 时 UCB 的 `mean + β·std`
还没有信息，到第 3 轮升到 0.0639。

**推论（影响起跑顺序）**：暖机那一步（`run_warmup_real_slime.sh`，dock 60 个分子）不是可选的性能优化，
而是**让 acquisition reward 在训练第一轮就有信号**的前提。跳过它，GRPO 的前若干轮拿到恒零 reward。

---

## 3. `vina_max_workers=32` 没有依据，实测饱和点在 8

PLAN §5 第 4 条要求「先量一个分子的真实耗时再定并发」。量了：

| 每轮评估数 | workers=1 | workers=8 | workers=32 |
|---|---:|---:|---:|
| 1（生产配置） | 5.4 s/轮 | 3.4 s/轮（**1.60×**） | 3.6 s/轮（1.52×） |
| 8 | 7.6 s/轮 | 5.3 s/轮（**1.43×**） | 5.4 s/轮（1.40×） |

**两点结论**：

1. **8 之后没有可测收益**，32 与 8 的差别（5.3 vs 5.4）落在噪声里。
   commit `5ca5efe9` 把它从 1 提到 32 的理由是「GH200 有 288 核」——
   **核数不是约束，每轮 dock 的分子数才是**。32 个 worker 里大部分没活干。

2. **总加速只有 1.4–1.6×，说明 docking 不是单轮耗时的主导项**，这与 PLAN §5
   「docking 是真瓶颈」相反。原因是 `vina_exhaustiveness=1`——**生产配置里也是 1**，
   所以 docking 本身就很便宜，其余时间花在 GP 拟合、活性预测与候选生成上。

**可操作的含义**：日后嫌慢时，调 `vina_max_workers` 不是那个旋钮。要先量出剩下那 60–70%
的时间花在哪里，再决定动谁。

---

## 3b. 训练栈 → 评测栈的进程边界已端到端验通（并补上一个会让 P0 白跑的缺件）

训练时 reward **不在训练进程里算**：`bridge.py` 起一个子进程
`task_python -m ldm_rl.real_env_worker`，走 JSON-lines stdio，并显式剥掉
`LD_LIBRARY_PATH` / `CUDA_HOME` / `CUDA_VISIBLE_DEVICES`。这个边界是代码写死的。

**缺件**：`bridge.py:87-89` 是

```python
task_python = str(spec.real.get("task_python")
                  or "/mnt/data0/ys/LDM/tasks/small_molecule/.venv/bin/python")
```

而 `gen_episodes_runs.sh` 把 `config.real_kwargs` 整块烤进 episode，**站点 config 里
原本没有 `task_python`**。于是四个 run 的 episode 全都缺这一项，训练时会回退到作者机器
的路径。已补进 `config_real.isambard.json` 并重新生成四个 episodes 文件。

**危害形态**：那个回退值在别的机器上依然是一个"合法"的字符串，所以它不报「配置缺失」，
而是在 reward 那一步报下游错误。加上 `remote_env.py:41` 把 worker 的 `stderr` 设成
`DEVNULL`，父进程只能看到一句 `task-venv worker exited unexpectedly`——**真实死因被丢掉了**。
（我自己第一次跑探针就撞上这个：忘了 export `PYTHONPATH`，worker 起来就
`ModuleNotFoundError` 退出，而父进程什么都不说。手动跑一次 worker 才看到原因。）

**端到端验证结果**——子进程起得来、协议往返通、reward 真的在另一个解释器里算出来：

| | round 0 | round 1 |
|---|---|---|
| 进程内（§1 的 E2） | vina −3.395 / act 5.9705 | vina −3.898 / act 5.9003 |
| **经子进程边界** | **vina −3.395 / act 5.9705** | **vina −3.898 / act 5.9003** |

两条路径**逐位一致**，所以远程路径不只是"能跑"，而是"算的是同一件事"。

**建议回馈上游**：`remote_env.py` 的 `stderr=subprocess.DEVNULL` 改成捕获并在
`RuntimeError` 里带出最后若干行。训练跑几小时后 worker 挂掉时，这是唯一的线索。

---

## 3c. 现成的 G12D 活性模型是旧版流程的产物，E9 需要一对流程匹配的评测器

E9 要「训 G12D → 评 **G12C（迁移）** + **G12D（同分布）**」。两侧各用一个活性模型打分，
所以两个模型必须由同一套流程产出，否则性能差里同时含**真实的迁移难度**与
**两个评测器自身的精度差**，分不开。

按 `best_g12d_model_metadata.json` 逐项对齐参数（IC50 / 只留精确关系 / 直接测定 /
seed 714 / test 0.2）跑出 G12C 之后，对不上的地方暴露出来了：

| | 仓库自带的 G12D | 当前脚本跑出的 G12C |
|---|---|---|
| splits | random · scaffold · **source_assay** · **assay_family** | random · scaffold · **assay** · **document** |
| 选中的 split | `scaffold` | `document` |
| 选中的模型 | `ensemble_nn_ridge_rf` | `ensemble_nn_ridge_lgbm` |
| 候选模型数 | 9 (`cpu_models`) | 28 |

选模型的规则是固定优先级 `(document, scaffold, assay, random)` 取第一个可用的
（`train_g12c_qsar.py:1311-1319`）。自带的 G12D **没有 document split**，所以落在 scaffold；
我们跑出来的有，所以落在 document。**这不是参数没对齐，是两份流程不同**——
split 定义与模型名册都变了，靠调 flag 对不齐。

**做法**：用**当前这份脚本**把 G12D 也重训一遍（`code/train_g12d_matched.sh`），
E9 用这一对流程匹配的评测器做主结果；仓库自带的那个保留，用于与作者的数对账。
代价很小——G12C 那次约 7 分钟。

G12C 结果（`results/g12c_qsar_20260901T010923Z/`）：

| 模型 | split | n_train | n_test | RMSE | MAE | R² | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| **ensemble_nn_ridge_lgbm** | document | 1364 | 482 | **0.6848** | 0.5259 | 0.2282 | 0.6248 |
| char_tfidf_ridge | document | 1364 | 482 | 0.6905 | 0.5349 | 0.2153 | 0.5826 |
| morgan_lightgbm | document | 1364 | 482 | 0.7021 | 0.5341 | 0.1887 | 0.6037 |

### 3c-1 选模型的规则会在一个只有 1 个测试分子的 split 上选（已修）

跑匹配版 G12D 时暴露的。各 split 的留出规模：

| split | G12C n_test | G12D n_test |
|---|---:|---:|
| assay | 406 | 376 |
| **document** | 482 | **1** ← 退化，占 0.1% |
| random | 370 | 141 |
| scaffold | 373 | 144 |

`select_best_model`（`train_g12c_qsar.py:1311`）按固定优先级
`(document, scaffold, assay, random)` 取**第一个存在**的 split，**不看它有没有足够的
测试点**。于是 G12D 在只有 1 个测试分子的 document split 上选模型——每个模型的 RMSE
就是那一个残差，赢家是"碰巧离那一个点最近"的那个。而运行照样打印
`Best model: morgan_hist_gradient_boosting selected on document split` 并把产物存下来，
**失败在下游完全不可见**。

已修：优先级顺序保留，但先跳过留出行数少于 30 的 split，跳过时明确打印原因；
全都不达标时退到留出集最大的那个并警告"选出的模型未经排序"。
验证（直接用两次已有的 metrics 重跑选择）：G12C 不变（document，482 行）；
G12D 打印 `NOTE: skipping 'document' split (n_test=1 < 30)` 后落到 scaffold（144 行），
选中 `rdkit_desc_ridge`。

### 3c-2 但结论反过来了：**不要**用当前流程替换自带的 G12D 模型

在两者都支持的 scaffold split 上做统一标尺：

| | n_test | RMSE | 标签 SD | RMSE/SD | R² | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| G12C（当前流程） | 373 | 0.6112 | 0.930 | **0.657** | **0.547** | **0.729** |
| G12D（当前流程） | 144 | 0.5040 | 0.670 | 0.752 | **0.028** | **0.226** |
| G12D（**仓库自带**） | 609 | 0.8850 | — | — | **0.739** | **0.865** |

G12D 当前流程的 RMSE 看起来更小（0.504 < 0.611），但那只是因为它的标签散布更窄
（SD 0.670 vs 0.930）；**除以各自的 SD 之后 G12C 反而更好**。而 R² 与 Spearman 说得更
直接：**当前流程做出的 G12D 评测器几乎不能给分子排序**（R²≈0.03，Spearman 0.23）。

原因是数据量：自带的 G12D 用了 **3044 行**，当前流程的 G12D 只拉到 **702 行**（4.3 倍差距）。
所以自带的那个不是"旧版遗留"，而是**更强的评测器**，训练也正是用它。

**修正后的做法**：
- G12D 评测**继续用仓库自带的** `best_g12d_model.joblib`（训练用的也是它）。
- G12C 只能用当前流程（没有自带产物），Spearman 0.729，尚可。
- 我训的匹配版 G12D **不作评测器**，只作诊断：它给出"同一套流程在 G12D 上能做到多好"
  的下界，用来判断 E9 里 G12C 与 G12D 的差距有多少可能来自评测器质量而非迁移本身。

**留给陈奕杭的一个问题**：当前脚本对 G12D 只拉到 702 行，而自带产物的输入是 3044 行
（`g12d_public_ic50_exact_relation_dedup_train_direct_assays.csv`）。是 ChEMBL 查询口径
变窄了，还是当时用了别的取数路径？**如果 G12C 也受同样的口径影响，那 G12C 评测器
（现在 1846 行）同样有变强的空间**，而它是 E9 迁移那一侧的唯一标尺。

---

## 4. 两个文件系统的约束是互补的（决定了环境该建在哪）

| | 空间 | inode |
|---|---|---|
| `/home`（VAST） | **101G 硬配额，曾撞顶** | 893k / 10000k（8.9%，宽裕） |
| Lustre | 116.9T / 200T（宽裕） | **49.96M / 51.2M（97.6%，只剩 1.24M）** |

上一次安装失败的直接原因是 `/home` 撞顶，但**根因不是空间不够**：

`install_train_env.sh` 第 24 行把 conda 包缓存指向 `$TMPDIR/conda_pkgs_$USER` 以避开配额，
**却没有 `mkdir -p`**。conda 的 `pkgs_dirs` 是一条**回退链而不是强制指定**：首选目录建不出来时
它不报错，静默滑到下一个可写目录 `~/.conda/pkgs`——正好在配额里面。于是「避开配额」这个配置
一次都没生效。报错路径 `/home/.../.conda/pkgs/cmake-4.4.3...` 就是证据。

已修：建目录 + 验可写 + 打印实际落点，建不出来就停。

**腾空间的做法也随之改了**：上一次是把冷环境逐文件 rsync 到 Lustre——那是**把 7.5 万个 inode
从富余的一侧搬到紧缺的一侧**，正好搞反。改成 `tar --remove-files`（每个文件写进归档之后才解除
源端链接，与 rsync 同级的安全性），目标端是一个文件：

| 冷环境 | 文件数 | 归档后 Lustre inode |
|---|---:|---:|
| glm53-vllm | 77,247 | **1** |
| glm53-vllm_cu130_deprecated（Lustre 侧） | 45,421 | **1** |
| glm53-vllm_cu130_deprecated（/home 侧） | 59,456 | **1** |

冷的依据是逐一查证过的：在跑的 `6220558/6220580` 用的是 apptainer SIF
（`images/vllm-glm53-arm64-cu129.sif`），不是这几个 conda 环境；而 `ldm-nanogpt`
正被 `run_repro_arm.sh:54` 使用，**没有动**。

---

## 4b. 训练栈安装:CUDA 红线已过,编译段的三个坑

**CUDA 红线过了**（HANDOFF §1 的头号风险）：

```
torch 2.11.0+cu129  cuda 12.9  avail True
```

装 `sglang[all]` 会拖进 cu13 的 torch 与运行库（驱动 565 只到 CUDA 12.7，装上去
`torch.cuda.is_available()` 直接 False）。按作者的办法强制回退——`--force-reinstall
--no-deps` 换 `+cu129`，再把 `nvidia-*-cu13` 卸掉换 `-cu12`——在 aarch64 上一次成功。
`sglang-0.5.15.post1-cp312-cp312-manylinux_2_34_aarch64.whl` 确实存在，PLAN §2 的
预判成立，**交接文档列的头号风险有一半被预编译轮子消掉了**。

### 坑 1：节点默认 gcc 是 7.5.0，torch 要求 ≥ 9

flash-attn 第一次编译失败，真实错误是

```
torch/include/c10/util/C++17.h:14:2: error: #error
    "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
```

`gcc --version` → **7.5.0 (SUSE)**。这会让 **flash-attn / TE / torch_memory_saver
三段全挂**。集群有 `gcc-native/{12.3,13.2,14.2}`；取 **12.3**（CUDA 12.9 的 nvcc 对宿主
编译器支持到 GCC 13）。已写进脚本并加了版本硬检查。

### 坑 2：`module` 在非交互 shell 里根本没定义，`module load` 静默无效

第一次修的时候直接在 `srun bash -c` 里写 `module load gcc-native/12.3`，**加载"成功"
后 gcc 仍是 7**——`module` 是一个 shell 函数，只有登录 shell 才会定义它，非交互 shell 里
这条命令什么都没做，也不报错。实测：

| 起法 | `module load` 后的 gcc |
|---|---|
| `srun bash -c '...'` | **7** ← 静默无效 |
| `srun bash -lc '...'` | **12**（`/opt/cray/pe/gcc-native/12/bin/gcc`） |

修法：脚本内部显式 source lmod 初始化再 load，路径是
`$MODULESHOME/init/bash` = `/opt/cray/pe/lmod/lmod/init/bash`
（**不是**常见的 `/usr/share/lmod/lmod/init/bash`，我第一次就猜错了）。

### 坑 3：`| tail -30` 把真正的编译错误扔了

脚本原本是 `pip install -v ... flash-attn 2>&1 | tail -30`。编译失败时真实的
nvcc/g++ 错误在前面几百行，tail 只留下 Python 的包装 traceback
（`RuntimeError: Error compiling objects for extension`），**等于把病因丢掉**，
只能重跑一遍才看到 "GCC too old"。已改成全量写进单独文件，失败时自动 grep 出前 5 条
真实错误并打印日志路径。

与 §3b 的 `stderr=DEVNULL` 是同一类：**为了让输出好看而截断，代价是失败时无法诊断。**

### 坑 4：编译被 OOM 杀掉，因为它在给**四个**架构编

第二次编译（gcc 12 已就位）跑到一半被杀：

```
slurmstepd: error: Detected 1 oom_kill event in StepId=6217606.119
sacct: State=OUT_OF_MEMORY   MaxRSS=316758912K  (302 GB)
```

nvcc 的中间文件名给出了线索：`flash_bwd_hdim192_bf16_sm80.compute_100.cudafe1.gpu`、
`.compute_120.` —— **sm80、sm100、sm120 都在编**，而 GH200 实测 `compute_cap = 9.0`。
读 setup.py 才看到它认的变量：

```python
# flash-attn 2.8.3  setup.py:69-70
@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")
```

**是 `FLASH_ATTN_CUDA_ARCHS`。** 它不读 `TORCH_CUDA_ARCH_LIST`（脚本里设了 9.0，没用），
也没有 `FLASH_ATTENTION_DISABLE_SM80` 这个开关——那是我凭一般印象设的，**属于同一类
「配置到不了它要控制的路径」的错误，只是这次犯错的是我自己**。教训：给第三方构建加
环境变量之前，先在它的 setup.py 里 grep 一遍那个名字，两分钟的事。

**修好之后的实测对照**（同一台 nid010402，同样的 gcc 12）：

| | 架构数 | 并发 | 结果 |
|---|---:|---|---|
| 第一次 | 4（80/90/100/120） | 32 × 4 线程 | 13 分钟才到 `[1/73]`，MaxRSS 302 GB，**OOM 被杀** |
| 第二次 | **1**（仅 90） | 16 × 4 线程 | **14 分钟跑完 `[73/73]`，零错误**，节点内存峰值 158 GB |

核对方式不是「设了变量」而是读构建文件：`build.ninja` 里
`grep -oE "arch=compute_[0-9]+,code=sm_[0-9]+" | sort -u` 只剩一行
`arch=compute_90,code=sm_90`。

设 `FLASH_ATTN_CUDA_ARCHS=90` 之后编译量与峰值内存都降到 1/4。并发也从
`MAX_JOBS=32 + --threads 4`（最多 128 路并发编译）降到 `16 + --threads 2`，
srun 侧另加 `--mem=380G`。

**一个被我读错又纠正过来的推断**：一度以为节点上的 `/tmp` 与 `$TMPDIR` 是 tmpfs
（确实是，RAM 支撑）且已经快满，因而怀疑是我把 conda/pip 缓存指过去才吃光内存。
实际是我把 `df` 的列读错了——`172G 7.5G 164G 5%` 是「总量 已用 可用 使用率」，
只用了 7.5G，根本没满。**OOM 与缓存位置无关，纯粹是编译本身。**
（顺手清掉上次被杀留下的 1112 个 pip/nvcc 瞬时目录，7.5G → 1.7G。）

### 坑 5：cuDNN 的头文件与库也不在 `$CUDA_HOME` 下（同一个结构的第三次）

TE 段挂在 `fatal error: cudnn.h: No such file or directory`。

cuDNN **不来自 conda 的 CUDA 包**，而是 pip 包 `nvidia-cudnn-cu12`，头文件在
`site-packages/nvidia/cudnn/include/`，库在 `.../cudnn/lib/`；`cudnn_frontend`
是另一个 pip 包，头文件直接在 `site-packages/include/`。而
`transformer_engine_torch` 的编译按 `$CUDA_HOME` 找。

**还有一层**：pip 的 cuDNN 只发**版本化的 `libcudnn.so.9`，没有裸的 `.so`**——
那个包是给运行时 `dlopen` 用的，而链接器要的是 `-lcudnn` 能解析到的 `libcudnn.so`。
所以除了软链，还要补出裸名。已建：头 14 个、库 16 个（含裸 `.so`）、frontend 28 个。

**这是同一个结构的第三次**（CUDA 头在 `targets/sbsa-linux/include`；cuDNN 在
`site-packages/nvidia/cudnn/include`）。共同点是**运行时能用、编译时找不到**——
Python 的加载器知道那些路径，编译器只看 `-I`。
**判断法**：报 `fatal error: X.h: No such file` 时先问「X 是谁装的」，
如果是 pip 包而不是 conda 的 CUDA 包，它一定不在 `$CUDA_HOME` 下。

### 坑 6：卸掉 cu13 会连带拿走 `cutlass`，把整个 TE 挡住

TE 装完之后 `import transformer_engine.pytorch` 直接炸：

```
File ".../transformer_engine/pytorch/attention/dot_product_attention/backends.py", line 167
  from flash_attn.cute.interface import (...)
File ".../flash_attn/cute/interface.py", line 29
  import cutlass
ModuleNotFoundError: No module named 'cutlass'
```

顺着依赖链查下去：

```
sglang ──依赖──> flash-attn-4 (4.0.0b15) ──依赖──> nvidia-cutlass-dsl[cu13]
                                                          │
                                            提供顶层模块 cutlass
                                                          ↑
                        安装脚本为了让 torch 认得卡,卸掉 nvidia-cutlass-dsl-libs-cu13
```

`nvidia-cutlass-dsl` **4.5.2 只有 cu13 这一个 extra，没有 cu12**
（`Requires-Dist: nvidia-cutlass-dsl-libs-cu13==4.5.2; extra == "cu13"`），
所以「把 cu13 换成 cu12」这条路对它不成立——cu12 的 libs 轮子里只有共享库，
没有 `cutlass` 模块（实测解压 20 个文件，`python_packages/` 是空的）。

而 TE 的那处 import 是**条件式**的：

```python
try:
    fa_utils.fa4_version = PkgVersion(get_pkg_version("flash-attn-4"))
except PackageNotFoundError:
    flash_attn_func_v4 = None          # ← 优雅分支
else:
    from flash_attn.cute.interface import ...   # ← 需要 cutlass
```

即**只要分发包 `flash-attn-4` 存在就会走进去**。

**做法：卸掉 `flash-attn-4`**，而不是把 cu13 库装回来。理由是三条，不是一条：
1. FA4 是 CuTe DSL 版，面向 **Blackwell sm100**；GH200 实测 `cap=(9,0)`，用不上。
2. TE 与 sglang 两侧对它缺失**都有优雅分支**（TE 见上；sglang
   `jit_kernel/flash_attention_v4.py:9-12` 是 `try/except` 后置空，只在真正调用
   v4 kernel 时才抛）。
3. 装回 cu13 库正是作者的流程要消除的东西——那会让 `torch.cuda.is_available()` 变假。

卸载后复验了两件事（它与我们源码编的 flash_attn 2.8.3 **共用 `flash_attn` 顶层**）：
`flash_attn 2.8.3` 仍在，`TE 2.16.0 import OK`。

### 坑 7：`sgl-router` 的 Python 包在子目录

```
ERROR: git+https://github.com/zhuzilin/sgl-router.git@v0.3.2-9daabcd
       does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
```

那个仓库的**根是 Rust crate**，Python 绑定在 `bindings/python/`（maturin 项目，
分发名 `sglang-router`、模块 `sglang_router`）。URL 要加
`#subdirectory=bindings/python`。

### 坑 7b：`sgl-router` 还要 `protoc`，而 conda 的 `protobuf` 不含编译器

子目录修好之后 maturin 真的开始编了，卡在下一处：

```
Error: Custom { kind: NotFound, error: "Could not find `protoc`. ..." }
   （smg-mesh crate 要编 src/proto/gossip.proto）
```

**`conda install -c conda-forge protobuf` 装了不管用**——那是 Python 绑定，
`protoc` 二进制在 **`libprotobuf`** 包里（装上后 `libprotoc 35.1`）。
已加进 env 段，并在 router 段开头加了一句显式检查 + 打印版本，
免得再等 cargo 编到一半才发现。

### 坑 7b：router 的 Rust 依赖要 `protoc`，而 conda 的 `protobuf` 不含编译器

子目录修好之后 maturin 真的开始编了，卡在下一处：

```
Error: Custom { kind: NotFound, error: "Could not find `protoc`.
       If `protoc` is installed, try setting the `PROTOC` environment variable..." }
cargo:rerun-if-changed=src/proto/gossip.proto
```

`smg-mesh` crate 要 Protocol Buffers 编译器来编 `gossip.proto`。
**注意 conda-forge 的 `protobuf` 是 Python 绑定，不含编译器**——
`protoc` 二进制在 `libprotobuf` 里。已加进 env 段（`libprotoc 35.1`）。

同族的提醒：环境里已经有 `protobuf 7.36.1`（pip，Python 运行时），
它与 `protoc` 是两回事；只看 `pip list` 会以为已经装了。

### 坑 8：一个"检查"若不改变控制流，它就只是一行日志

`tms` 段的断言失败了，链条却照样跑到下一段——脚本没有 `set -e`，heredoc 的非零
退出被吞掉。已改成显式 `[ $? -eq 0 ] || exit 1`。

**但更值得记的是：那次断言本身是错的。** 它写作

```python
d = pathlib.Path(torch_memory_saver.__file__).parent
so = list(d.rglob('*.so'))     # ← 在包目录里找
```

而这个包的扩展模块装成 **site-packages 的同级模块**
（`torch_memory_saver_hook_mode_preload_cu12.abi3.so`），不在包目录里。
于是它对一次**成功**的安装报了 FATAL——构建日志明明写着
`torch_memory_saver-0.0.9.post1+slime-cp312-cp312-linux_aarch64.whl`（平台轮子），
`_cu12` 后缀还说明 `TMS_CUDA_MAJOR` 也生效了。

改成问包管理器要安装记录（`importlib.metadata.files()`），与文件落在哪无关。
这与 §5 的「import 成功不等于装上了」是同一枚硬币的两面：
**检查要问包管理器，不要问文件系统的某个猜测位置。**

### TE 的 backward 在 aarch64 上不 SIGSEGV（HANDOFF §7 的头号风险）

交接文档要求用 1.5B 冒烟来验这件事，但那要先转检查点、起 ray、起 sglang，
一次十几分钟，而且真挂掉时症状混在一堆分布式错误里。
`code/probe_te_backward.py` 用几秒钟的最小实验直接回答——构造 TE 的
`Linear` + `LayerNormMLP`，bf16 前向、反向、看梯度：

```
device: NVIDIA GH200 120GB  cap=(9, 0)
transformer_engine 2.16.0
前向 OK  out=(256, 512)  loss=0.099190
反向 OK  有梯度的参数张量 8 个  |grad| 求和 86.5874  输入梯度 0.2116
TE_BACKWARD_OK
```

**不 SIGSEGV，梯度有限且非零。** 这不替代 P0 的全链冒烟（那还要验 GRPO、
rollout、reward 的接线），但它把「TE 装错了」从待验清单里划掉了。
已接进 `run_p0_smoke.sh` 的起跑前检查。

### 顺带：`--gres=gpu:4` 下 step 内看到的设备号（实测，不是推理）

```
step 内 CUDA_VISIBLE_DEVICES=0,1,2,3   SLURM_STEP_GPUS=0,1,2,3
0..3 各 1-2 MiB(真空)
```

所以 `run_train_real_slime.sh` 写死的 `CUDA_VISIBLE_DEVICES=1,2,3`（actor 1 卡 +
sglang 2 卡）在 `--gres=gpu:4` 下**有效**；若只申请 3 张卡，设备 3 不存在，
而这类错误不会报错、只会让那一路指向不存在的卡。P0 因此申请 4 张。

---

## 4c. Ray 这一层的两个问题（P0 卡在这里，与 TE 无关）

### 一、`ray start` 返回 ≠ job 提交服务能收请求

第一次 P0 起跑后 6 分钟就结束，死在

```
RuntimeError: Request failed with status code 504: .
```

时间线是决定性的：

| 时刻 | 事件 |
|---|---|
| 07:30:31 | `ray start --head` 开始 |
| 07:30:48 | 报 `Ray runtime started`（用了 17 秒） |
| 07:30:54 | `ray job submit` 发出，**6 秒后**拿到 504 |

端口在听，后面的 dashboard agent 还没起。**作者机器上启动快，这 6 秒够用；
换台机器就是每次必挂**，而症状看起来像 Ray 的 bug 而不是竞态。

已给八个启动脚本加就绪等待——轮询 `/api/version` 直到 200，超时 120s 报明确原因。
实测这一步「等了 2s」就通过，与上面那 6 秒并不矛盾：**关键不是等多久，
而是等到确认为止，而不是等一个猜的秒数。**

### 二、就绪之后 `POST /api/jobs/` 本身超时

加了就绪等待之后仍然 504，这次等了 **5 分钟**（07:38:53 → 07:43:55）。
dashboard 的访问日志给出了确切位置：

```
07:38:53  'GET /api/version HTTP/1.1' 200      ← ray job submit 的版本检查
（之后没有任何完成的 POST /api/jobs/）
07:41:32  'GET /api/jobs/ HTTP/1.1' 200 160 bytes   ← 我手工跑的 ray job list，返回 []
```

即**创建作业的那个请求在 JobHead 子进程模块里超时了**。而同期
`gcs_server` 与 `raylet` 的日志一直在正常更新（07:42:38 仍有心跳），
JobHead 自身启动也无报错——**集群是好的，只有提交这条路不通**。

**做法：去掉这一层，而不是绕过它。** 在**单节点本地集群**上，job server
买不到任何东西：同一个进程 `ray.init` 连上已起的集群就能跑，而
`runtime_env` 里那三个变量（`PYTHONPATH` / `LD_LIBRARY_PATH` /
`CUDA_DEVICE_MAX_CONNECTIONS`）本来就已经在当前 shell 里 export 过，
本地起的 worker 直接继承。

加了 `SLIME_LAUNCH_MODE` 开关：不设或 `jobsubmit` 是作者原样，
`direct` 走 `RAY_ADDRESS=auto python3 train.py`。站点配置选 `direct`，
**上游默认不受影响**。

---

## 4d. 一次 OOM 把 openai 包弄成半装，几小时后才显形

P0 走通 direct 路径、真正进入 `train.py` 之后，死在

```
ModuleNotFoundError: No module named 'openai._models'
```

调用链是 `train.py → slime.utils.arguments → sglang.srt.server_args →
sglang.srt.entrypoints.openai.protocol → openai.types.responses`。

查下去：`site-packages/openai/` **只剩子目录**
（`auth` `_extras` `helpers` `lib` `providers` `resources` `types` `_utils` `_vendor`），
顶层 `.py` 一个都没有，包括 `_models.py` 与 `__init__.py`。
多半是 flash-attn 那次 OOM 把 pip 打断在写文件的中途。

**为什么这类损伤特别难自查**：

| 检查手段 | 结果 |
|---|---|
| `pip list` | `openai 2.6.1` —— 完全正常 |
| `import openai` | 不报错 |
| `pip check` | 通过（它只查依赖关系） |
| 实际使用 | 踩到 `openai._models` 才炸，而那已经是训练启动到一半 |

**一次 OOM 的损伤可以延迟几小时才显形，并且看起来像一个完全无关的模块错误。**

已强制重装修好。另写 `code/verify_env_integrity.py`：按每个包的
`dist-info/RECORD`（安装时写下的文件清单）逐个核对文件是否真的在。
**扫了 302 个有安装记录的包，openai 是唯一损伤，其余全部完整。**
已接进 P0 起跑前检查。

这与 §4b 坑 8 的 `tms` 是同一条原则的两个方向——**以包管理器的安装记录为准**：
那次是别去文件系统的猜测位置找 `.so`，这次是拿记录去验文件在不在。

至此 P0 的起跑前检查有四层，每一层都是被一次真实失败逼出来的：

```
文件在不在  →  包真装上了没有  →  文件与安装记录一致吗  →  TE 的 backward 真跑得动吗
   (缺件)        (命名空间包假阳性)      (半装的包)              (装得上 ≠ ABI 对得上)
```

---

## 5. 启动脚本的配置到不了它要控制的路径

`slime_launch/*.sh` 里 `REPO_ROOT` / `MEGATRON_ROOT` / `CONDA_PREFIX` / `MODEL_HF` / `CONFIG`
全是**直接赋值**，外部 export 会被无声覆盖。改成 `${VAR:-<作者原值>}` 之后：作者机器上不设环境变量
行为完全不变，本机 `source site_env.sh` 才生效。命令行参数里的 `--hf-checkpoint` / `--ref-load` /
`--prompt-data` 也一并由 `REPO_ROOT` 与新增的 `HF_MODELS` 推出。

改完逐脚本求值验证过，五个脚本的全部变量都解析到本机路径（不是只看改没改）。

**订正 PLAN §4 的一处依赖**：E7（暖机 GP）在 PLAN 里写成只依赖 E2，实际上
`run_warmup_real_slime.sh` 需要 slime + 1.5B 的 torch_dist 检查点——它是 LLM 驱动的 rollout
去提分子再 docking。所以 **E7 依赖 E3 与 E5**，不能提前做。

---

## 6. 还没做的

| 阶段 | 内容 | 卡在哪 |
|---|---|---|
| E3 | 训练栈安装 | 进行中（sglang aarch64 轮子已确认存在） |
| E4 | **P0**：1.5B GRPO 冒烟，验 TE backward 不 SIGSEGV | 等 E3 |
| E5 | HF → Megatron torch_dist 转换 | 等 E3 |
| E6 | **P1**：9B real 极小 count 冒烟 | 等 E4+E5 |
| E7 | 暖共享 GP（60 个分子） | 等 E3+E5（依赖已订正） |
| E8 | **P2**：R1–R4 × 3–5 seed 真训练 | 等全部 |
| E9 | 离线评测 G12C + G12D | 等 E8；`best_g12c_model.joblib` 可并行训 |
