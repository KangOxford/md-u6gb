# A · 纯 Mamba3 baseline 代码地图（sigma-0）

**调研时间**：UTC 2026-08-11T20:45Z
仓库根：`/lus/lfs1aip2/projects/public/u6gb/sigma-0`

---

## 1. 模型入口：registry

`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/registry.py`

| 项 | 值 | 行号 |
|---|---|---|
| 注册表 `_DEFINITIONS` | `mamba3` / `s5` / `gdn` / `kda` / `transformer` / `nsa` | 41–48 |
| 纯 mamba3 注册名 | **`"mamba3"`**（也是 `DEFAULT_ARCHITECTURE`） | 16, 42 |
| token 模式常量 | `TOKEN_MODE="26tok"`, `TOKENS_PER_MESSAGE=26` | 17–18 |
| `mamba2` **显式拒绝** | `ValueError("architecture 'mamba2' is intentionally not part of this model zoo")` | 60–64 |
| 选择器解析（architecture/ssm_type/model_type 三路合一，冲突报错） | `resolve_architecture` | 69–120 |
| mamba3 构建分支（读 `mamba3_*` 超参 → `init_Mamba3SSM`） | `build_backbone` | 188–223 |
| **attention 分支**（transformer/nsa，`wrapper_policy="internal"`，自带残差） | | 290–350 |
| carry 元数据按宽度重算 | `recurrent_carry_for_width` | 159–185 |
| **attention cache 按 `d_book` 缩减头数** | `attention_config_for_width` | 131–156 |

> 注意：`mamba2` 在本仓库被主动拒绝，指令 (1) 要求「mamba3 而非 mamba2」与仓库现状一致，无需额外工作。

公共 re-export（新代码从这里 import，实现留在 `s5` 以保 ckpt 参数名兼容）：
`src/base_model/models/__init__.py:7-18`、`src/base_model/models/mamba3.py:3`

---

## 2. Mamba3 实现与超参

| 文件 | 绝对路径 | 行数 | 作用 |
|---|---|---|---|
| Flax 层 | `src/s5/mamba3.py` | 526 | `Mamba3SSM`(:70)、`init_Mamba3SSM`(:505) |
| 纯 JAX 数值核 | `src/s5/mamba3_jax.py` | 376 | `mamba3_ssd_chunked_jax`(:209)、`apply_rope`(:84)、`angle_dt_cumsum`(:123) |
| Triton 核入口 | `src/s5/mamba3_ops.py` | 1125 | `mamba3_siso_triton` |
| Triton 核实现 | `src/s5/mamba3_kernels/` | — | `siso_fwd/siso_bwd/siso_combined/angle_dt/utils` |
| CUDA FFI | `src/s5/state_scan_ops.py` | — | `state_scan_cuda`，替换 phase 4/5/6 |

### 超参对照（`mamba3.py:85-97`）

| Flax 字段 | 默认 | CLI flag | env |
|---|---|---|---|
| `H` | — | `--d_model` | `D_MODEL` |
| `d_state` | 128 | `--mamba3_d_state` | `MAMBA3_D_STATE` |
| `expand` | 2 | `--mamba3_expand` | `MAMBA3_EXPAND` |
| `headdim` | 64 | `--mamba3_headdim` | `MAMBA3_HEADDIM` |
| `chunk_size` | 64 | `--mamba3_chunk_size` | `MAMBA3_CHUNK_SIZE` |
| `rope_fraction` | 0.5 | `--mamba3_rope_fraction` | `MAMBA3_ROPE_FRACTION` |
| `A_floor`/`dt_min`/`dt_max` | 1e-4/0.001/0.1 | 无 | — |
| `use_triton`/`use_cuda` | False/False | 同名 | 同名大写 |
| `tp_size` | 1 | `--tp_size` | `TP_SIZE` |
| `step_rescale` | 1.0 | — | **被接受但忽略**(:83,:97) |

派生量：`eff_headdim=min(headdim, expand*H)`(:102)、`n_heads=d_inner_target//eff_headdim`(:103)、`num_rope_angles=int(d_state*rope_fraction)//2`(:120-123)。

### Mamba3 相对 Mamba2 的独有机制

**本仓库没有 mamba2 实现**（registry:60-64 拒绝）。Mamba3 独有：

| 机制 | 行号 |
|---|---|
| RoPE 旋转状态（`split_tensor_size`/`num_rope_angles`） | `mamba3.py:119-123` |
| `in_proj` 多出 `trap` 与 `angles` 两组通道（Mamba2 只有 z,x,B,C,dt） | `mamba3.py:127-131`，切分 :190-197 |
| 角度累积 + RoPE 施加 | `mamba3_jax.py:84-121, 123-148, 253-256, 282-284` |
| 梯形离散化 `gamma/shifted_gamma/scale` | `mamba3_jax.py:259-269`；RNN 版 `mamba3.py:443-458` |
| BCNorm（B、C 各一个 RMSNorm） | `mamba3.py:146-147`，施加 :218-219 |
| QK-dot 对角 skip | `mamba3_jax.py:278-280`；开关 `mamba3.py:473-480` |
| **无 depthwise conv1d**（Mamba2 有短卷积，GDN 有 `use_conv`） | 全文无 conv 参数 |
| TP 无关的全尺寸 `out_norm`/`out_proj`（ckpt 跨 tp_size 互换） | `mamba3.py:45-67,152-159,255-302` |

> ⚠️ **无 conv1d 对混合设计有直接影响**：Nemotron 的 NoPE 依赖 Mamba 的 `conv1d(k=4)` 提供局部相对位置。Mamba3 没有 conv1d，但**有 RoPE（`rope_fraction=0.5`）内建在 SSM 状态里**，位置信息由此提供。详见设计文档。

---

## 3. 模型组装链（插 attention 的关键）

```
node_wrapper.sh:536  python run/base_model/runtime/train.py
  └─ src/lob/train.py:103-136   create_lobster_prediction_dataset(...)
  └─ src/lob/train.py:150       init_train_state(...)
       └─ src/lob/init_train.py:348-354   build_backbone(args) → backbone.layer_factory
       └─ src/lob/init_train.py:355-359   token_mode != '26tok' → raise
       └─ src/lob/init_train.py:409-430   merging=='padded' 分支
            partial(BatchPaddedLobPredModel, ssm=ssm_init_fn, n_fused_layers=args.n_layers, ...)
            └─ src/lob/lob_seq_model.py:723  BatchPaddedLobPredModel = nn.vmap(PaddedLobPredModel)
                 └─ PaddedLobPredModel.setup (:412-475)
```

`PaddedLobPredModel.setup` 建三段栈（`src/lob/lob_seq_model.py`）：

| 子模块 | 层数 | 行号 |
|---|---|---|
| `message_encoder = StackedEncoderModel(use_embed_layer=True, vocab_size=d_output)` | `n_message_layers`=2 | 416–431 |
| `book_encoder = LobBookModel(pre/post)` | 1 + 1 | 434–448 |
| `fused_s5 = StackedEncoderModel(...)` | `n_fused_layers = args.n_layers` = 6 | 453–472 |
| `decoder = nn.Dense(d_output)` | — | 474 |

**真正堆 N 层的类**：`src/s5/seq_model.py` 的 `StackedEncoderModel`(:7)，`setup` 在 **58–80**：

```python
self.layers = [ SequenceLayer(ssm=self.ssm, ...,
                              use_moe=self.use_moe and (i % self.moe_every_n == 0), ...)
                for i in range(self.n_layers) ]
```

### 可插拔性结论：**硬编码同一种层**

- **没有**按 index 选择 mixer 类型的抽象，所有层共享同一个 `self.ssm` factory（`seq_model.py:59`、`lob_seq_model.py:157,170`）。
- 唯一已存在的「按 index 分支」是 MoE：`seq_model.py:70` `use_moe = self.use_moe and (i % self.moe_every_n == 0)`，`moe_every_n` 默认 2(:42)。**这是现成的 index hook 模板，插 attention 照抄即可。**
- `SequenceLayer` 已有 mixer 类型分派雏形（`src/s5/layers.py`）：

| 已有机制 | 行号 |
|---|---|
| 解嵌套 `functools.partial` 探测 `is_transformer` | 44–49 |
| transformer/nsa 直接透传（自带 Pre-LN + residual），不走外层 norm/GLU/residual | 96–97 |
| 推理按层类型分派 `__call_inference__` vs `__call_rnn__` | 161–163, 172–175 |
| `prefill`（只有 attention 层需要 KV cache） | 141–149 |
| `initialize_carry` 按 `ssm_type` 造 carry（mamba3 四元组/gdn/transformer/nsa） | 218–262 |

### 插 attention 的三个改造点

| # | 位置 | 改法 |
|---|---|---|
| **1** | `seq_model.py:58-80` | 接受 **per-layer ssm factory 列表**（或 `layer_kinds: tuple[str,...]`），而非单个 `self.ssm` |
| **2** | `seq_model.py:142-152` + `lob_seq_model.py:679-705` | `initialize_carry` 目前对 N 层统一用同一 `ssm_type`；混合栈须按层分别造 carry（recurrent 四元组 vs KV cache） |
| **3** | `registry.py:188-354` | 目前只返回**一个** `BackboneBuild`；hybrid 需同时构造 mamba3 + transformer 两个 build，复用 `attention_config_for_width`(:131-156) 处理 `d_book` 头数缩减。`BackboneDefinition.wrapper_policy`(external/internal) 与 `is_attention`(:36-38) 已存在可直接用 |
| 附 | `seq_model.py:106-113` prefill | 已按层收集 cache 列表，mamba3 层返回 `None`(`layers.py:149`)，**无需改** |

---

## 4. 训练配置

> **重要区分**：`configs/train/*.yaml` **不是超参文件**，只是 launcher plan（package/entrypoint/legacy_batch/sbatch_*/env_* 透传）。真超参在 batch 脚本 preset 表 + 提交 env。

| 配置 | 路径 |
|---|---|
| 纯 mamba3 + SP500（声明式） | `configs/train/mamba3_sp500.yaml`（32 节点 / `MODEL_PRESET=75m` / `EPOCHS=40` / 26tok） |
| mamba3 冒烟 | `configs/train/mamba3_smoke.yaml`（1 节点 GOOG 2022 三天） |
| model zoo 冒烟 | `configs/train/model_zoo_smoke_mamba3.yaml` |
| **实际跑出 baseline 的配方** | `/lus/.../tasks/sp500_mamba3_35m_20260805T030348Z/scripts/launch_attach_train.sh` |

### 尺寸预设（`run/base_model/train_full_autoreg.batch:229-250`）

| preset | D_MODEL | N_LAYERS | BLOCKS | SSM_SIZE_BASE | PER_GPU_BSZ |
|---|---|---|---|---|---|
| 75m | 1024 | 6 | 16 | 1024 | 10 |
| 360m（默认） | 2048 | 24 | 32 | 2048 | 2 |

`ARCHITECTURE=mamba3` 且无 `MODEL_PRESET` 时**直接报错退出**(:252-268)，防止静默落到 360M。

### baseline 完整配置（33.6M 档）

| 类别 | 设置 |
|---|---|
| 架构 | `ARCHITECTURE=mamba3`, `SSM_TYPE=mamba3` |
| 尺寸 | `D_MODEL=640`, `N_LAYERS=6`, `BLOCKS=20`, `SSM_SIZE_BASE=640` → 实测 **33,610,439** 参数 |
| Mamba3 | `d_state=128, expand=2, headdim=64, chunk_size=64, rope_fraction=0.5, use_triton=False`（BSZ>1 时 OOM） |
| 隐式层 | `n_message_layers=2, n_book_pre_layers=1, n_book_post_layers=1`（`runtime/train.py:98-104` 默认） |
| 序列 | `TOKEN_MODE=26tok`, `MSG_SEQ_LEN=500` → **13000 token/样本**；`book_depth=500` |
| Batch | `PER_GPU_BSZ=4` × 4 GPU × 4 节点 = **global 64**；`GRAD_ACCUM_STEPS=1, TP_SIZE=1, HIERARCHICAL=True` |
| 优化器 | `OPT_CONFIG=muon, MUON_LR=0.01, SSM_LR_BASE=8.0e-4`（µP：5e-4×1024/640），`WEIGHT_DECAY=0.005, LR_FACTOR=1, p_dropout=0.0` |
| LR schedule | warmup 线性 `WARMUP_END=0.01`（320 步）→ cosine；`COSINE_STEPS=32000` 覆盖 total_steps（生效点 `src/lob/train.py:431-433`） |
| 长度 | `EPOCHS=1, MINI_EPOCHS=1, CURTAIL_EPOCHS=32000`（单位是**步**），`MAX_JOB_HOURS=5.0` |
| Checkpoint | `CHECKPOINT_EVERY=3000`，`save_before_timeout_minutes=30` |
| 固定项（`node_wrapper.sh:537-560`） | `--prenorm=True --batchnorm=False --bidirectional=False --merging=padded --activation_fn=half_glu1 --C_init=trunc_standard_normal --clip_eigs=True --use_book_data=True --book_transform=True --masking=none --token_mode=26tok` |
| W&B | `oxford-lob/sp500-mamba3-35m`，run `30nkkohd` |

有意偏离原 8 节点配方两处（脚本注释写明）：不设 `LOCAL_STEPS_K=10`（DiLoCo 在 4 节点收益小）；用墙钟而非步数封顶。

---

## 5. 训练入口与提交

| 角色 | 路径 | 关键行 |
|---|---|---|
| Plan CLI（dry-run 打印 sbatch） | `run/base_model/train_base_model.py` | 全 34 行 |
| **主 sbatch** | `run/base_model/train_full_autoreg.batch` | 1445 行；`#SBATCH` 6–15；preset 229–250；**srun 573–581**；auto-resume 594–730 |
| per-node wrapper | `run/base_model/node_wrapper.sh` | squashfs 挂载 360–414；python 启动 **536**–640 |
| Python 训练脚本 | `run/base_model/runtime/train.py` | argparse：`--architecture`:129, `--n_layers`:104, `--cosine_anneal`:213, `--warmup_end`:215 |
| 训练主循环 | `src/lob/train.py` | 数据 103–136；`init_train_state` 150；LR 424–445；`train_epoch` 658–664 |
| 模型构建 | `src/lob/init_train.py` | 348–354, 409–430 |

**4 节点两条起法**

1. 常规排队：`ARCHITECTURE=mamba3 MODEL_PRESET=75m GPUS_PER_NODE=4 sbatch --nodes=4 --time=23:59:00 run/base_model/train_full_autoreg.batch`
2. **附着常驻 4 节点占位链**（baseline 实际用法）：占位作业 `/lus/.../tasks/u6gb_16_nodes_daily_log/four_node_chain.sbatch`；attach 改写见 `launch_attach_train.sh:35-38`（sed 把 `srun --nodes=$NNODES` 改成 `srun --jobid=<ID> --overlap --nodes=$NNODES`），须先清 `SLURM_*` 继承变量(:97-108) 与残留 squashfuse 挂载(:60-85)。记账器 `record_submission.py`，账本 `submissions.jsonl`/`events.jsonl`。

---

## 6. 数据与 tokenizer

| 项 | 值 |
|---|---|
| 数据根（SquashFS 月度分片） | `/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs`（`shard_YYYY-MM.squashfs`） |
| 覆盖 | 2022-01 … 2025-12，**48 个月**；`TRAIN_DATE_RANGE=2022-01-01,2025-12-31` ← **与指令 (6) 完全一致** |
| 8 只股票 | `GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD` ← **与指令 (6) 完全一致** |
| 挂载 | `node_wrapper.sh:360-414`（`SQUASHFS_MULTI_MODE=1`，逐月 squashfuse，48 挂载点拼成 `DATA_ROOT`） |
| loader 注册 | `src/lob/dataloading.py:422-425` → `"lobster-prediction"` |
| loader 入口 | `src/lob/dataloading.py:134`；train loader :297 |
| 多 ticker 发现 | `src/lob/lobster_dataloader.py:410 discover_ticker_files`（**必须有 index.json，glob fallback 已禁用**，防 MDT 风暴） |
| 采样器 | `dataloading.py:22 UniformTickerDistributedSampler`（本 baseline `UNIFORM_TICKER_BATCHES` 未开） |

### tokenizer：26 tok/message，定长，非 BPE

`src/lob/encoding.py`

| 项 | 行号 | 值 |
|---|---|---|
| `TOK_LENS` | 351 | `(1,1,3,2,1,3,2,3,3,2,2,3)`，sum=**26** |
| 位布局 | 349–350 | `evt:0, dir:1, price:2-4, size:5-6, dt_s:7, dt_ns:8-10, time_s:11-12, time_ns:13-15, price_ref:16-18, size_ref:19-20, time_ref:21-25` |
| 编解码入口 | 163–164, 228–229 | `encode_msg_26` / `decode_msg_26` |
| `Vocab` 特殊 token | 262, 271 | MASK=0, HIDDEN=1, NA=2, START=3 |
| 字段词表 | 277–286 | time(1000)+event_type(4)+size_digit(100)+price(1000)+sign(2)+direction(2) |
| **vocab size** | 注释 279 | **2112** |
| `seq_len` | `lobster_dataloader.py:945` | 500 × 26 = **13000** |
| `n_classes` | `lobster_dataloader.py:1635-1636` | 2112 |

仓库另有 `encoding_1tok/22tok/23tok/24tok/26tok.py` 与 `numeric_bpe.py`，但 `init_train.py:355-359` 硬性只允许 `26tok`，`node_wrapper.sh:598` 也写死 `--token_mode=26tok`。BPE 是未接入主线的实验路径。

---

## 7. 已训好的 baseline checkpoint

**仓库无 `latest_checkpoint.json` breadcrumb**（全仓 grep 零命中）。索引载体：
- `/lus/lfs1aip2/projects/public/u6gb/live_jobs.md:5426-5453`（本 baseline 完整记录）
- 每个 ckpt 目录下 `metadata/_ROOT_METADATA`（Orbax custom_metadata，含全部超参）
- 程序化查找器 `run/base_model/find_model_zoo_checkpoint.py:18 find_checkpoint(root, architecture, d_model, n_layers)`

### ★ 最新纯 mamba3 baseline

| 项 | 值 |
|---|---|
| **绝对路径** | `/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints/j5877859_30nkkohd_5877859` |
| **step** | **32001**（另存 3000…30000 共 11 个中间点） |
| 元数据 | `architecture=mamba3, ssm_type=mamba3, model_type=ssm, token_mode=26tok, d_model=640, n_layers=6, blocks=20, ssm_size_base=640, msg_seq_len=500, mamba3_d_state=128/expand=2/headdim=64/chunk_size=64/rope_fraction=0.5, merging=padded, prenorm=true, restore=null` |
| 参数量 | **33,610,439** |
| 数据 | SP500 SquashFS 48 月，8 tickers，26tok |
| W&B | `https://wandb.ai/oxford-lob/sp500-mamba3-35m/runs/30nkkohd` |
| 训练日志 | `/lus/.../tasks/sp500_mamba3_35m_20260805T030348Z/logs/train_attach_20260805T032251Z.log` |
| **LOB-Bench**（job 5924045） | `/lus/.../tasks/sp500_mamba3_35m_20260805T030348Z/bench_20260806T160429Z_j5924045/summary.json`：**ks=0.1064, l1=0.1629, wasserstein=0.2088**，21 features，3136 条 GOOG Jan-2026 冻结评测池 |

### 其它 mamba3 checkpoint（对照）

| 路径 | 最新 step | 配置 | 用途 |
|---|---|---|---|
| `checkpoints_selftrain/j5705912_b30675li_5705912` | **69378** | d_model=1024, n_layers=6, blocks=16, 500msg, 26tok（75M 档） | `sigma0-selftrain`；longctx 的 restore 源 |
| `checkpoints/j5877859_ariremic_5877859` | 仅 metadata | 同 35m 配方 | 同 job 早期失败 run |
| `checkpoints/j5944477_pk9dr3ez_5944477` | 61 | d_model=1024, **msg_seq_len=2000** | `sigma0-longctx` |
| `checkpoints/j5944477_{gf50nzy1,wqy53g6y}_5944477` | 仅 metadata | **msg_seq_len=4000** | longctx，未产步 |

> **hybrid 应对齐的 baseline = `j5877859_30nkkohd_5877859 @ 32001`**：唯一有完整 LOB-Bench 数字、且用 8 股票 2022–2025 全量数据训到收敛的纯 mamba3 run。
