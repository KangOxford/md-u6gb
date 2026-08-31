# 在 Isambard-AI 上跑 LDM / LDM-RL —— 实操手册

**给谁看**:要在这台机器上跑 LDM 系列实验的人(含刚拿到账号、还没登录过的)。
**写于** 2026-08-31,内容全部来自本机实跑,不是照抄文档。
**维护者**:junming(u6gb 项目下 `tasks/large-discovery-model/`)

---

## 0. 三十秒版本

已经跑通并可直接复用的东西:

| 东西 | 位置 | 状态 |
|---|---|---|
| LDM 仓库 | `tasks/large-discovery-model/Large-Discovery-Models` | ✅ b5dab16a |
| 三套 Python 环境 | `~/envs/ldm-venv`、`ldm-causal`、`ldm-nanogpt` | ✅ 建好,含 torch 2.9.1+cu128 |
| GLM-5.3-Flash 推理服务 | `glm53_flash/code/serve_vllm_sif.sh` | ✅ 一条命令起,TP4 |
| nanoGPT 训练数据 | `data/autoresearch_cache`(788M) | ✅ 含 8192 BPE 词表 |
| MLS-Bench 钉版 | `vendor/MLS-Bench` @ cfd57a7e | ✅ sha 核验过 |
| A/B 对照实验 | `glm53_flash/code/launch_repro_selfhosted.sbatch` | ✅ 正在跑 |

**新人最快路径**:跳到 §2 起环境 → §4 跑 mock(五分钟,不用 GPU 不用 LLM)→ §5 起真跑。

---

## 1. 接入集群

### 1.1 登录

不是"用邮箱登录"。流程是:BriCS 发账号 → 上传 SSH 公钥 → ssh 进登录节点。

```bash
ssh <你的用户名>@login.isambard.ac.uk    # 具体主机名以 BriCS 邮件为准
```

登进来先确认落在哪:`hostname` 会显示 `login40`/`login41`/`login42` 之类。
**登录节点是本地的**——tmux server、后台进程都只在你登录的那台上,换一台就看不见。

### 1.2 项目盘与配额

代码和数据都在 Lustre 上:

```
/lus/lfs1aip2/projects/public/u6gb/        ← 项目根(也可从 /projects/public/u6gb 进)
  tasks/large-discovery-model/             ← LDM 这条线的全部东西
```

查配额(容量 200T,**inode 5120 万,这个更容易先满**):

```bash
lfs quota -h -p $(lfs project -d /lus/lfs1aip2/projects/public/u6gb | awk '{print $1}') /lus/lfs1aip2
```

⚠️ **Python venv 不要建在项目盘**。一个带 torch 的 venv 有几万个文件,几个就把 inode 吃掉。
建到 `$HOME/envs/` 下(HOME 在另一个文件系统,15P)。

### 1.3 🚨 Lustre 禁令(违反会连累全组)

2026-05-08 全组作业被管理员挂起过一次,原因是元数据请求把 MDT 压垮。以下命令**禁止执行**:

```bash
ls -R / ls -lR <path>        # 任何递归 ls
find /projects ... / find / ...   # 大范围 find
du -sh <大目录>               # 递归 stat
tree <lustre 路径>
ls -1td <ckpt_dir>/*          # 用 ls 找最新 checkpoint
watch ls / while true; do ls; done
```

替代做法:`lfs find <窄路径> -name "*.py"`、直接 `cat` 已知路径、找最新 checkpoint 读
`latest_checkpoint.json` 这种小文件。**不确定某个命令会不会打元数据,就别试,问一下。**

登录节点还禁止:编译、GPU 计算、重数据预处理、常驻 agent。
允许:改文件、`squeue`/`sacct`、轻量 CPU 脚本(<30 分钟、<16GB、顺序读写)。

---

## 2. 环境

三套环境已建好,**直接用,不用重装**:

| 环境 | 用途 | 装了什么 |
|---|---|---|
| `~/envs/ldm-venv` | LDM 共享层、数据增强 | ldm_tts + openai 客户端 |
| `~/envs/ldm-causal` | causal_discovery 任务 | 上面 + pgmpy |
| `~/envs/ldm-nanogpt` | nanoGPT 任务(**要 GPU**) | 上面 + torch 2.9.1+cu128 |

要自己重建的话(比如换任务):

```bash
cd tasks/large-discovery-model/Large-Discovery-Models
UV_PROJECT_ENVIRONMENT=$HOME/envs/<名字> uv sync --project tasks/<任务名>
# 需要真训练的任务再加 --group train
```

`uv` 在 `/projects/public/u6gb/.local/bin/uv`。

**torch 版本别乱升**:节点驱动 565(CUDA 12.7),cu128/cu129 轮子能用,**cu130 装上去
`torch.cuda.is_available()` 直接是 False**——PyPI 上 torch 的默认轮子已经切到 CUDA 13 了,
装之前先看 `+cuXXX` 后缀。

---

## 3. 起 LLM 端点(LDM 的"提案模型")

LDM 的每一轮都要调 LLM 生成候选,所以先得有一个 OpenAI 兼容端点。
群里说"一定要和最牛的基础模型比较"——本机已经跑通 **GLM-5.3-Flash**(320B 总参 /
18B 激活,MIT,原生多模态,1M 上下文),权重已下好在
`tasks/large-discovery-model/models/GLM-5.3-Flash`(FP8,306G,62 分片)。

### 3.1 起服务

```bash
# 需要一个有 4 张空卡的节点(GLM 走 TP4)
bash tasks/large-discovery-model/glm53_flash/code/attach_serve_sif.sh <jobid> <node>
```

脚本会先**逐卡验空**(>2GB 就拒绝起),然后用 apptainer 拉起官方镜像。
起来后是 `http://<node>:8383/v1`,模型名 `GLM-5.3-Flash`。

装载 306G 权重约 1 分钟(页缓存热)到 4 分钟,之后 DeepGEMM 要预热 ~1000 个内核 +
抓 CUDA graph,**首次总共约 10 分钟**。看到 `Application startup complete` 才算好。

### 3.2 为什么是 apptainer 而不是 pip

试过六轮源码构建,结论是**只有官方镜像这条路走得通**:

- PyPI 的 vllm 发行版**没有 Glm5Next 架构**,会静默退回通用后端,然后死在 KDA 层参数上;
- 实现在未合并的 PR#53906(93 文件、含 CUDA 内核改动),纯 Python 打补丁做不到;
- 自己编 PR 分支会连撞 conda CUDA 布局、CUTLASS 版本、宿主编译器四道坎。

镜像已经拉好转成 SIF:`glm53_flash/images/vllm-glm53-arm64-cu129.sif`(9.5G)。

### 3.3 换别的模型

任何 OpenAI 兼容端点都行(vLLM / SGLang / 外部 API)。LDM 只认三个环境变量:

```bash
export LLM_BASE_URL=http://<node>:8383/v1   # 注意结尾是 /v1,不是 /chat/completions
export LLM_MODEL_NAME=GLM-5.3-Flash
export LLM_API_KEY=dummy                     # 本地服务随便填
# nanoGPT 任务还认历史变量名:TTS_LLM_URL / TTS_LLM_MODEL / TTS_LLM_API_KEY
```

⚠️ **不要把真 API key 写进 YAML 或提交到仓库**。committed 的 real 配置里
`llm-url/api-key` 都是 `null`,就是为了让凭证只走环境变量。

---

## 4. 先跑 mock(五分钟,不烧 GPU 不调 LLM)

任何新任务、新环境,先过这一关:

```bash
cd tasks/large-discovery-model/Large-Discovery-Models
python scripts/validate_tasks.py --task <任务名>
$HOME/envs/ldm-venv/bin/python scripts/run_ldm_tts.py config/<任务名>/mock.yaml --dry-run
$HOME/envs/ldm-venv/bin/python scripts/run_ldm_tts.py config/<任务名>/mock.yaml
```

产物落 `tasks/<任务名>/runs/mock/`,里面有 `result.json`、`events.jsonl`、
`experiment_contract.json`(契约快照)等。

**开数据采集**(生成 SFT 语料用):

```bash
export LDM_DATA_COLLECTION_ENABLED=1
export LDM_DATA_COLLECTION_DIR=/绝对路径/data/generated/<campaign 名>
```

⚠️ **必须用绝对路径**。runner 会 `pushd` 进任务目录再执行,相对路径会落到
`tasks/<任务名>/` 底下而不是你以为的仓库根。

---

## 5. 跑真实验

### 5.1 提交作业的铁律

**每次 `sbatch` 之前先跑 `gtop`**。自己名下有空卡就 attach 上去,不要去排队:

```bash
gtop                                    # 看此刻每张卡的物理占用
squeue -u $USER -t PENDING -o "%.10i %.28j %.5D %R"   # 同时看排队的
```

判空只认**逐卡显存**:1-4 MiB 才是真空;`gtop` 头行的 idle 计数把 held(有人预留
显存)也算进去,不能作依据。attach 写法:

```bash
srun --overlap --jobid=<分配> -w <节点> --ntasks=1 --gres=gpu:1 --cpu-bind=none bash <脚本>
```

### 5.2 定时评测必须独占 GPU

nanoGPT 任务的评测是**跑满 300 秒训练然后报 val_bpb**。这意味着:

> 被别人抢卡 ⇒ 同样 300 秒跑更少 token ⇒ bpb 变差。

**GPU 争抢会把「哪组邻居更闲」伪装成「哪组方法更好」。** 做对照实验时,
`sbatch` 必须带 `--exclusive`,每条 campaign 独占一张卡。这条是血泪:第一次
A/B 试跑就是在共卡下起的(邻居铺满了 12 张卡),只能整个作废重跑。

### 5.3 服务和实验要在同一个作业里

第二个血泪:把 GLM 服务留在分配甲、实验提交成作业乙,分配甲到期后服务消失,
乙的 6 条 campaign 全部在 `glm_health=000` 下**空跑了一小时才结束**——脚本只是
打印健康码,没有据此停下。

现在的正确形态见 `glm53_flash/code/launch_repro_selfhosted.sbatch`:

```
一个作业,3 节点独占
  节点0: GLM 服务(TP4 占满 4 卡)
  ↓ 等健康检查通过(最多 30 分钟,不通过直接 exit,不空跑)
  节点1: A1 B1 A2 B2  (每条一张卡)
  节点2: A3 B3
```

每条 campaign 起跑前还会**再查一次健康码,非 200 拒跑**。

### 5.4 提交与监控

```bash
sbatch code/launch_repro_selfhosted.sbatch
squeue -j <jobid>
# 作业消失后必须查退出码,"不在 squeue" ≠ 成功
sacct -j <jobid> --format=JobID,State,ExitCode -nP
```

---

## 6. 做对照实验时最容易搞错的两件事

这两条是我在真跑里踩出来的,写代码之前先看:

### 6.1 `breadth × depth` 是真评测次数,不是"候选池大小"

直觉上会以为 breadth/depth 控制"生成多少候选交给代理打分"(便宜),
**实际上每一个都要跑一次真训练**(每次 6 分钟)。

```
best_of_n:  每轮真评测 = breadth × depth
  breadth=4 depth=4, 20 轮 → 345 次真评测(35 小时)
  breadth=1 depth=1, 20 轮 →  45 次真评测(4.5 小时)
```

**如果两组的 breadth/depth 不同,预算就差了 7.7 倍**,比出来的差距会被
"多跑了 7 倍评测"解释掉,和方法本身没关系。论文提交的配置是 breadth=1 depth=1
(实测「100 次迭代中 99 个进入真训练」),两组都该照这个。

### 6.2 LDM 的机制开关是 `--acquisition-feedback`,默认是**关的**

真正把"代理的预测与不确定性"喂回提案模型的,是这个参数:

| 参数 | 默认 | 作用 |
|---|---|---|
| `--acquisition-feedback` | **none** | `brief`/`verbose` 才把 GP 引导写进提案 prompt |
| `--acquisition-feedback-probes` | 96 | 用多少局部探针总结 GP 引导 |
| `--surrogate-mode` | lcb | 采集函数(lcb/ucb/mean/ei) |

所以 A/B 对照的正确形态是:**两组 breadth/depth/迭代数/起点/模型全部相同,
只有 `acquisition-feedback` 一个是 brief、一个是 none。**

### 6.3 别自己设 `CUDA_VISIBLE_DEVICES`

`srun --gres=gpu:1` **已经把一张卡隔离给这个 step 了**——step 内部看到的永远是设备 0。
如果你在 launcher 里再写 `CUDA_VISIBLE_DEVICES=1/2/3` 想"分卡",那三条指向的是
**不存在的设备**,训练会在 11 秒内失败并返回哨位值 `1e+09`,而且**不报错、不中断**,
一路跑完给你一份全是失败评测的结果。

分卡靠 Slurm(`--gres=gpu:1 --exclusive`),不靠环境变量。

### 6.4 前置检查要发"真实形状"的请求

`/health` 返回 200 **不代表你要用的那条路通**。实测踩过:服务起来了、健康码 200,
但起服时漏了 `--tool-call-parser glm47 --enable-auto-tool-choice`,而 `operation_tool`
生成器走的是 tool calling —— 每个候选都吃 **400 Bad Request**,30/30 外层候选
`generation_error`,整个 campaign 白跑。

规律:**前置检查查的是"进程在不在",真正会坏的是"这条具体路径通不通"。**
所以 `run_repro_arm.sh` 现在有两道检查,都是发真实请求:

```bash
# 1. 健康码,非 200 exit 9
# 2. 带 tools 字段真发一次 chat/completions,返回里有 error 就 exit 10
```

写任何新的 runner 时照这个做:**探针的形状要和真实调用一样**。

### 6.5 契约会硬校验,别绕过它

任务的 `experiment.json` 用 `locked_args` 锁死方法和预算,配置对不上会直接报
`Config violates experiment contract profile`。**不要改 committed 的 profile**,
按它的 schema 加一个新的:

```json
"profiles": {
  "my_new_profile": {
    "description": "...",
    "budget": {"outer_iterations": 40, "warmup_evaluations": 5},
    "locked_args": {"method": "best_of_n", "iterations": 40, "warmup": 5}
  }
}
```

契约的 sha256 会随每次运行快照进 run 目录,改动可追溯——这是好事,别躲着它走。

---

## 7. 加一个新任务(比如 CUDABench)

群里提到拿 [CUDABench](https://github.com/CUDA-Bench/CUDABench) 当新任务测 LDM。
仓库有现成的脚手架,**不用改共享代码**:

```bash
python scripts/scaffold_task.py cuda_bench \
  --description "Optimize CUDA kernel source against the pinned CUDABench evaluator."
```

生成 14 个文件,自己要写的是四个:

| 文件 | 内容 |
|---|---|
| `resources/upstream_contract.json` | 上游 URL、40 位 commit、task_path、关键文件 sha256 |
| `experiment.json` | 指标角色、评测设置、单候选限额、预算、profile |
| `core/evaluator.py` | 调外部评测器,返回 objective |
| `core/proposals.py` / `candidate.py` | 候选生成、解析、规范化、校验 |

`ldm_task/procedure.py` 保持薄:解析 CLI → `describe_ldm_task` → `run_campaign`。
样板抄 `tasks/ai4bio_mutation_effect_prediction/`(唯一"外部钉版 benchmark +
qualified"的任务)。完整规则见 `skills/register-ldm-task/SKILL.md`。

写完必过五步验证:

```bash
python scripts/validate_tasks.py --task cuda_bench
uv run --project tasks/cuda_bench pytest tasks/cuda_bench/tests
python scripts/check_task_dependencies.py config/cuda_bench/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/cuda_bench/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/cuda_bench/mock.yaml
```

**CUDABench 特有的一点**:它的评测本身要占 GPU 编译+跑 kernel,和 §5.2 同理——
评测若带计时,必须独占卡。

---

## 8. LDM-RL 接进来时要注意什么

(LDM-RL 代码还在两位同学手上,这里只写与本机相关的接口约束)

1. **提案模型端点复用 §3**,不用自己起服务。RL 训练如果要 rollout,注意 GLM
   TP4 会吃掉一整个节点的 4 张卡,规划节点数时算进去。
2. **采集到的轨迹已经是现成的 RL 数据**:`ldm_ir.jsonl` 是 ldm-2.0 IR
   (含 action / search_state / outcome),`augment.py` 可以再加专家 reasoning。
   格式契约见 `data/SCHEMA.md`。
3. **训练脚本必须自带 checkpoint + resume**。这台机器上作业有 walltime,
   没有 resume 的长训练一超时就全没了(有过 168 GPU-hours 打水漂的先例)。
   checkpoint 写节点本地 `$TMPDIR`,作业结束再 rsync 回 Lustre;
   频率按"恢复一次浪费 <10% 时间"定,一般 ~15 分钟一次。
4. **`GRAD_ACCUM_STEPS` 不要写默认值**。声明有效批量,由节点数推导 K 并硬校验——
   换节点数忘了改 K 不会报错,只会训出一个不可比的模型。

---

## 9. 资源与时间

- **GPU hours 9 月 10 日到期**,ICLR 前大约两周窗口。
- 单次 nanoGPT 评测 ≈ 6 分钟(300 秒训练 + 装载),所以:
  - 45 次评测的一条 campaign ≈ 4.5-5.5 小时
  - 3 种子 × 2 组 = 6 条并行 ≈ 同样 5 小时(每条一张卡)
  - 论文口径的 120 次评测 ≈ 12 小时/条
- **一个节点 4 张 GH200(96G/卡)**,GLM 服务吃 4 张,实验每条 1 张。
- 排队时:短时限(≤2h)更容易被 backfill 塞进去;16 节点这种大块可能等几小时。

---

## 10. 遇到问题先看这里

| 症状 | 多半是 |
|---|---|
| `torch.cuda.is_available()` 是 False | 装了 cu130 轮子,换 cu128/cu129 |
| 服务起不来,`NVCC compilation failed` | 容器内 TMPDIR 指向宿主未绑定路径,钉 `TMPDIR=/tmp` |
| 服务报 `Mamba cache blocks (512)` | GLM 的 KDA 层限制,加 `--max-num-seqs 256` |
| 服务报 `pe_dim must be 64 for fp8_ds_mla` | 去掉 `--kv-cache-dtype fp8`(与 NoPE-MLA 不容) |
| campaign 里 `val_bpb=1e+09` | 该候选代码跑挂了(哨位值)。**这很常见,约一半** |
| **每次**评测都是 1e+09 且只跑了十几秒 | 自己设了 `CUDA_VISIBLE_DEVICES`,见 §6.3 |
| 候选全是 `generation_error` | 服务缺 tool-calling 标志,见 §6.4 |
| `Config violates experiment contract` | 见 §6.3,加新 profile 而不是改老的 |
| 作业"跑完了"但没结果 | 查 `sacct` 退出码,别信"不在 squeue 就是成功" |
| 采集目录空的 | `LDM_DATA_COLLECTION_DIR` 用了相对路径,见 §4 |

**`val_bpb=1e+09` 值得单独说**:实测 LLM 生成的候选里**约一半跑不通**
(A 组 45 有效 / 40 失败,B 组 45 有效 / 45 失败)。框架会重试到拿够预算内的有效
评测,所以预算对齐仍然成立,但这个失败率本身是个值得报告的数字——它衡量的是
"提案模型生成可运行代码的能力",换更强的模型这个数应该降。

---

## 附:本机已有的东西一览

```
tasks/large-discovery-model/
├── Large-Discovery-Models/          # 上游仓库 b5dab16a
│   ├── config/<任务>/               # mock.yaml / real*.yaml
│   ├── tasks/<任务>/                # 6 个任务
│   └── data/generated/              # 采集产物(gitignored)
├── models/GLM-5.3-Flash/            # 306G FP8 权重
├── data/autoresearch_cache/         # nanoGPT 数据 + BPE 词表
├── vendor/MLS-Bench/                # 钉版 cfd57a7e
├── glm53_flash/
│   ├── code/serve_vllm_sif.sh       # 起服务
│   ├── code/attach_serve_sif.sh     # attach 起服(带验空)
│   ├── code/launch_repro_selfhosted.sbatch   # 服务+实验同作业
│   ├── code/ng_A_ldm.yaml / ng_B_llmonly.yaml # A/B 对照
│   ├── images/*.sif                 # vLLM 官方镜像 9.5G
│   ├── logs/                        # 全部运行日志
│   └── results/RUNLOG.md            # 逐条时间线,踩坑全记录
└── docs/RUNBOOK_LDM_ON_ISAMBARD.md  # 本文
```

有问题直接问,或者看 `glm53_flash/results/RUNLOG.md`——那里面是逐条的时间线,
每个坑什么时候踩的、怎么定性的、怎么修的都在。

---

## 11. 在 565 驱动上跑 CUDA 13（已实测打通，2026-08-31）

节点驱动是 **565.57.01 = CUDA 12.7**，而 CUDA 13 的应用要求驱动 ≥580。驱动是内核模块，
没有 root 改不了。但 NVIDIA 对**数据中心卡**（GH200/H100 符合）提供
**forward compatibility**：装一份用户态的 `libcuda.so`，让应用加载它而不是系统那份。

### 裸机（自己的 venv）

```bash
# 一次性:取 compat 库解到 HOME,系统一个字节不动
cd ~/envs && mkdir -p cuda13-compat && cd cuda13-compat
curl -sL -o compat.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-compat-13-0_580.95.05-0ubuntu1_arm64.deb
ar x compat.deb && tar xf data.tar.*

# 用时挂在最前面
export LD_LIBRARY_PATH=~/envs/cuda13-compat/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
```

实测 `torch 2.13.0+cu130`：不挂 `cuda_available=False`，挂上 `True` + 4 卡认到 +
2048² 矩阵乘结果正确。现成环境在 `~/envs/cu13-test`。

### 容器（apptainer）—— 这里有个坑

**vLLM 官方 cu130 镜像自带 `/usr/local/cuda/compat/libcuda.so.580.82.07`，
但开箱即用仍然是 `cuda_available=False`。** 因为 `apptainer --nv` 注入的是宿主 565 的
驱动库，容器自己的 compat 目录不在 `LD_LIBRARY_PATH` 上。必须显式指定：

```bash
apptainer exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 ...
```

不写这一行，任何人拿到这个镜像都会得出「CUDA 13 在这台机器上用不了」的错误结论。
落地脚本见 `glm53_flash/code/serve_vllm_sif_cu130.sh`（会自动探测容器有没有自带
compat，没有就把宿主那份绑进去）。

### 实测结果（vLLM cu130 镜像 + GLM-5.3-Flash）

| 项 | 结果 |
|---|---|
| 服务启动 | ✅ 570–590 秒就绪 |
| 对话 | ✅ 答案正确 |
| tool calling | ✅ 正确产出 `set_numeric(...)` —— LDM 的 `operation_tool` 走这条路 |
| LDM campaign 端到端 | ✅ `best_score 0.990808`（真实 bpb，优于基线 1.0193） |

### 版本对应与注意

`cuda-compat-13-N` 的包版本号是它提供的**用户态驱动版本**：13-0→580、13-2→595、13-3→610。
装 13-0 最保守。

**还没验证的**：多机 NCCL 在 compat 下的行为；Flash-Attention 等自定义扩展的编译对齐。
遇到诡异的底层报错时，先把 compat 摘掉对照跑一次，能很快分辨是不是它引起的。

### 一个布局教训

cu130 的第一次 LDM 测试四次评测全是哨位值，`stderr` 里是 `CUDA out of memory`——
**服务和评测训练放在了同一个节点**，服务 TP4 吃满 82 GiB/卡，训练申请不到显存。
服务与评测必须分节点，这和 §5 「定时评测必须独占 GPU」是同一件事的两个面。
