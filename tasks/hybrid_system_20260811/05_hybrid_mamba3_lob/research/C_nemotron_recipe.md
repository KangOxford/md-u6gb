# C · NVIDIA Nemotron Nano 9B v2 精确混合层配方

**调研时间**：UTC 2026-08-11T20:40Z
**证据来源**（全部为本地已下载官方文件，固定 revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`）：

| 文件 | 用途 |
|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/02_nemotron_nano_9b_v2_inference/model/config.json` | 全部超参 |
| `.../model/modeling_nemotron_h.py` | 前向实现、norm、init |
| `.../model/configuration_nemotron_h.py` | pattern 断言 (`:196-197`) |
| `.../model/model.safetensors.index.json` | 张量形状**独立交叉验证** |
| `.../model/README.md` | 官方 model card |
| `.../HYBRID_ARCHITECTURE_REPORT.md` §3.2 | 本地分析（非官方论述） |

---

## 1. 层模式串

```
M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-
```

长度 **56** = `num_hidden_layers`（`configuration_nemotron_h.py:196` 断言强制相等，`:197` 限字符集为 `[*-M]`）。

| 字符 | 层类型 | 类名 | 权重键 |
|---|---|---|---|
| `M` | Mamba-2 序列混合 | `NemotronHMamba2Mixer` | `in_proj/conv1d/A_log/D/dt_bias/norm/out_proj` |
| `*` | Attention 序列混合 | `NemotronHAttention` | `q_proj/k_proj/v_proj/o_proj` |
| `-` | MLP 通道混合 | `NemotronHMLP` | `up_proj/down_proj` |

> **移植时最容易搞错的一点**：Nemotron-H **不是**「每 block = mixer + FFN」。它把 token mixing 与 channel mixing 拆成**各自独立的 residual 层**，每层只有一个 mixer + 一个 pre-norm（`modeling_nemotron_h.py:741-791`）。

---

## 2. 比例与精确 index

| 类型 | 层数 | 占 56 层 | 占 31 个**序列混合层** |
|---|---:|---:|---:|
| Mamba-2 `M` | **27** | 48.2% | 87.1% |
| MLP `-` | **25** | 44.6% | — |
| Attention `*` | **4** | 7.1% | **12.9%** ← 正确基数 |

**Attention 精确 index（safetensors 交叉验证：仅这 4 层含 `q_proj`）**

| 0-based | 相对深度 | 在 31 个混合层中序号 | 相对混合深度 |
|---:|---:|---:|---:|
| **14** | 25.0% | 第 9 | 25.8% |
| **21** | 37.5% | 第 13 | 38.7% |
| **30** | 53.6% | 第 18 | 54.8% |
| **39** | 69.6% | 第 23 | 71.0% |

**分布是「非均匀、居中偏上」，不是等间距**：

| 区段 | index | 长度 | 组成 | Mamba 数 |
|---|---|---:|---|---:|
| 前置纯递归 | 0–13 | 14 | `M-M-M-MM-M-M-M` | 8 |
| ★ Attn 1 | 14 | 1 | `*` | — |
| 间隔 1 | 15–20 | 6 | `-M-M-M` | 3 |
| ★ Attn 2 | 21 | 1 | `*` | — |
| 间隔 2 | 22–29 | 8 | `-M-M-M-M` | 4 |
| ★ Attn 3 | 30 | 1 | `*` | — |
| 间隔 3 | 31–38 | 8 | `-M-M-M-M` | 4 |
| ★ Attn 4 | 39 | 1 | `*` | — |
| 后置纯递归 | 40–55 | 16 | `-M-MM-M-M-M-M-M-` | 8 |

四条硬观察：① 前 25% 深度无 attention，第一个 attention 前有 8 个 Mamba；② 后 29% 深度无 attention，最后一个 attention 后仍有 8 个 Mamba；③ 第 0 层是 Mamba、第 55 层是 MLP，两端都不是 attention；④ attention 间距（按混合层计）为 3/4/4，密集区约 **1 attn : 3.5 Mamba**。

### ASCII 层堆叠

```
                     norm_f RMSNorm(4480) → lm_head (131072×4480, 不共享)
  L55  [ - ] MLP        ────┐
  L54  [ M ] Mamba-2        │
  L53  [ - ] MLP            │
  L52  [ M ] Mamba-2        │
  L51  [ - ] MLP            │  后置纯递归区
  L50  [ M ] Mamba-2        │  16 层 / 8 Mamba / 8 MLP
  L49  [ - ] MLP            │  最后一个 attention 之后
  L48  [ M ] Mamba-2        │  仍有 8 个 Mamba 做局部整理
  L47  [ - ] MLP            │
  L46  [ M ] Mamba-2        │
  L45  [ - ] MLP            │
  L44  [ M ] Mamba-2  ←┐ MM │
  L43  [ M ] Mamba-2  ←┘    │
  L42  [ - ] MLP            │
  L41  [ M ] Mamba-2        │
  L40  [ - ] MLP        ────┘
 ═L39═ [ * ] ATTENTION ★4   40Q/8KV/d=128, NoPE      ← 69.6%
  L38  [ M ] Mamba-2
  L37  [ - ] MLP
  L36  [ M ] Mamba-2        间隔 3: 8 层 / 4 Mamba
  L35  [ - ] MLP
  L34  [ M ] Mamba-2
  L33  [ - ] MLP
  L32  [ M ] Mamba-2
  L31  [ - ] MLP
 ═L30═ [ * ] ATTENTION ★3                            ← 53.6%
  L29  [ M ] Mamba-2
  L28  [ - ] MLP
  L27  [ M ] Mamba-2        间隔 2: 8 层 / 4 Mamba
  L26  [ - ] MLP
  L25  [ M ] Mamba-2
  L24  [ - ] MLP
  L23  [ M ] Mamba-2
  L22  [ - ] MLP
 ═L21═ [ * ] ATTENTION ★2                            ← 37.5%
  L20  [ M ] Mamba-2
  L19  [ - ] MLP            间隔 1: 6 层 / 3 Mamba（最短）
  L18  [ M ] Mamba-2
  L17  [ - ] MLP
  L16  [ M ] Mamba-2
  L15  [ - ] MLP
 ═L14═ [ * ] ATTENTION ★1                            ← 25.0%
  L13  [ M ] Mamba-2   ────┐
  L12  [ - ] MLP           │
  L11  [ M ] Mamba-2       │
  L10  [ - ] MLP           │  前置纯递归区 14 层 / 8 Mamba
  L09  [ M ] Mamba-2       │  ★ 全模型无 RoPE，
  L08  [ - ] MLP           │    这 8 层 Mamba/conv 是
  L07  [ M ] Mamba-2  ←┐MM │    attention 唯一的位置信息来源
  L06  [ M ] Mamba-2  ←┘   │
  L05  [ - ] MLP           │
  L04  [ M ] Mamba-2       │
  L03  [ - ] MLP           │
  L02  [ M ] Mamba-2       │
  L01  [ - ] MLP           │
  L00  [ M ] Mamba-2  ────┘
                     embeddings (131072×4480)
```

---

## 3. Attention 细节：GQA + **NoPE**

| 参数 | 值 | 来源 |
|---|---:|---|
| `num_attention_heads` (Q) | **40** | config.json |
| `num_key_value_heads` | **8** | config.json |
| Q:KV 组比 | **5** | `modeling:852` |
| `head_dim` | **128**（显式给出，非 4480/40=112） | config.json |
| q_proj | 4480 → 5120 | index.json |
| k/v_proj | 4480 → 1024 | index.json |
| o_proj | 5120 → 4480 | index.json |
| `attention_bias` | **false** | config.json |
| `sliding_window` | **null**（全局全注意力） | config.json |
| causal | `is_causal=True` | `modeling:854,897` |

### RoPE：**没有**（NoPE）

`modeling_nemotron_h.py` 全文无 `rotary` / `apply_rotary_pos_emb` / `RotaryEmbedding`。`NemotronHAttention.forward` 里 `position_embeddings` 那一行是**被注释掉的 TODO**（`:864`）；`position_ids` 被接收但从未使用。`max_position_embeddings=131072` 只用于 cache 尺寸推断。非层级权重只有 `embeddings.weight` / `norm_f.weight` / `lm_head.weight`，**没有任何绝对位置表**。

位置信息完全由 Mamba-2 提供：① SSM 递归严格因果、按序累积，天然编码顺序；② 每个 Mamba 层内 depthwise `conv1d(kernel=4)` 提供局部相对位置；③ attention 层前永远有足够多的 Mamba 层（第一个之前有 8 个）。

> **移植硬含义：绝不能把 attention 放在第 0 层。**

---

## 4. Mamba-2 细节

| 参数 | config 键 | 值 | 备注 |
|---|---|---:|---|
| d_state | `ssm_state_size` | **128** | |
| n_heads | `mamba_num_heads` | **128** | |
| headdim | `mamba_head_dim` | **80** | |
| d_inner | 派生 | **10240** | `=128×80`（`modeling:295`），**不是** expand×hidden |
| expand | `mamba_expand` | 2（名义） | **代码未使用**；有效 expand = 10240/4480 = **2.2857** |
| n_groups | `n_groups` | **8** | |
| conv_kernel | `conv_kernel` | **4** | depthwise |
| chunk_size | `chunk_size` | **128** | config.json 顶层覆盖了代码默认 256（本地实例化验证 `c.chunk_size==128`） |
| conv_dim | 派生 | **12288** | `=10240+2×8×128`，与 `conv1d.weight[12288,1,4]` 一致 |
| in_proj 输出 | 派生 | **22656** | `=10240(z)+12288(x,B,C)+128(dt)`，与权重 `[22656,4480]` 一致 |
| dt 范围 | `time_step_min/max/floor` | 1e-3/0.1/1e-4 | |
| conv bias | `use_conv_bias` | **true** | 全模型唯一有 bias 的模块 |
| proj bias | `use_bias` | **false** | |

---

## 5. Norm 与残差

| 项 | 结论 | 位置 |
|---|---|---|
| Norm | **RMSNorm**，方差在 **fp32** 里算 | `modeling:724-739` |
| eps | `1e-5` | config.json |
| Pre/Post | **Pre-norm**：`res=h; h=norm(h); h=mixer(h); out=res+h` | `modeling:769-790` |
| 每层 norm 数 | **1 个**（无 post-mixer norm） | |
| 最终 norm | `norm_f` RMSNorm(4480) | `modeling:1293,1408` |
| **QK-norm** | **没有** | safetensors 层 14 只有 `{q,k,v,o}_proj` |
| Mamba 内 gating | **有**：`MambaRMSNormGated`，group_size=10240/8=1280，被 `z` 分支门控。全模型唯一的 gate | `modeling:263-278,339` |
| MLP | **无 GLU**，只有 `up_proj→act→down_proj` 两个矩阵，激活 **`relu²`**（不是 SwiGLU），宽度 **3.5×d** | `modeling:795-814` |
| `residual_in_fp32` | false | config.json |
| `rescale_prenorm_residual` | **true**：init 时 `out_proj`/`down_proj` × `1/√(2·n_layer)` | `modeling:1145-1148` |
| `tie_word_embeddings` | false | config.json |

参数量自检：embed 587M + lm_head 587M + 27×147.5M + 25×140.5M + 4×55M ≈ **8.88B** ✓

---

## 6. 「为什么这样排」：**本地资料未载**

| 问题 | 本地是否有官方说明 |
|---|---|
| attention 位置选择理由（为何 14/21/30/39） | **未载** |
| attention 比例选择理由（为何 4 层） | **未载** |
| 消融数据（retrieval 能力 vs 线性成本） | **未载** |

model card (`model/README.md:51`) 只有描述性陈述无理由：

> "The model uses a hybrid architecture consisting primarily of Mamba-2 and MLP layers combined with just four Attention layers."

理由外链两篇 arXiv（Nemotron-H `2504.03624`、Nemotron Nano 2 `2508.14444`），**两篇 PDF 本地均未下载**。本地 `HYBRID_ARCHITECTURE_REPORT.md:83,154,158` 的相关说法是**该报告作者的分析**，不是 NVIDIA 论述，引用须标明。

---

## 7. 缩放到 12–24 层小模型

### 7.1 先纠正一个会算错的比例

直接用 **4/56 = 7.1%** 是错的：56 层里 25 层是 MLP-only，**不做任何 token mixing**。若目标模型用常规「1 block = mixer + FFN」，正确对标量是 attention 占**序列混合层**的比例：

$$r=\frac{4}{27+4}=\frac{4}{31}=0.129\approx\frac{1}{7.75}$$

| 缩放口径 | 公式 | L=12 | L=16 | L=20 | L=24 |
|---|---|---:|---:|---:|---:|
| ❌ 按总层数 7.14% | `round(0.0714·L)` | 1 | 1 | 1 | 2 |
| ✅ **按混合层 12.9%** | `round(0.129·L)` | **2** | **2** | **3** | **3** |
| 参考 Jamba2 3B (2/28) | 7.14% | 1 | 1 | 1 | 2 |
| 参考 Kimi Linear (7/27) | 25.9% | 3 | 4 | 5 | 6 |

**Rule A（对标 Nemotron）**：$k=\max(1,\ \mathrm{round}(0.129\cdot L))$

### 7.2 位置换算：按深度带插值，不用固定间隔

Nemotron 的 4 个 attention 落在相对深度 **{0.250, 0.375, 0.536, 0.696}**。抽象成 **[0.26, 0.71] 深度带**内线性等分：

$$d_i=0.26+(0.71-0.26)\cdot\frac{i}{k-1},\quad i=0,\dots,k-1\quad(k\ge2)$$
$$d_0=0.48\quad(k=1,\ \text{取四个深度的质心})$$
$$\mathrm{idx}_i=\mathrm{clamp}(\mathrm{round}(d_i\cdot L),\ 2,\ L-2)$$

自检：L=56, k=4 → 0-based 14/24/34/40，与真值 14/21/30/39 平均偏差 1.75 层，形状吻合。

**换算表（0-based block index，L = mixer 块数）**

| L | k | attention block index |
|---:|---:|---|
| 12 | 2 | **3, 9** |
| 14 | 2 | **4, 10** |
| 16 | 2 | **4, 11** |
| 18 | 2 | **5, 13** |
| 20 | 3 | **5, 10, 14** |
| 22 | 3 | **6, 11, 16** |
| 24 | 3 | **6, 12, 17** |
| 28 | 4 | **7, 11, 15, 20** |

L=12 展开（`R`=递归块，`A`=attention 块）：

```
block:  0  1  2  3  4  5  6  7  8  9 10 11
type:   R  R  R  A  R  R  R  R  R  A  R  R
        └──┬──┘                 └─┬─┘
      前 3 块纯递归              尾 2 块纯递归收束
      = NoPE 的位置信息来源
```

### 7.3 移植硬约束（违反其一即显著掉点）

| # | 约束 | Nemotron 依据 |
|---|---|---|
| **C1** | **block 0 不得为 attention**；第一个 attention 前至少 `max(2, ceil(0.20·L))` 个递归块 | 无 RoPE；Nemotron 第一个 attention 前有 8 个 Mamba |
| **C2** | **末层不得为 attention**；最后一个 attention 后至少 2 个递归块 | Nemotron 最后 attention 在 L39，之后 8 Mamba + 8 MLP |
| **C3** | 不允许两个 attention 相邻 | 最小间隔 6 层（3 Mamba） |
| **C4** | attention 间至少隔 3 个递归块 | 间隔为 3/4/4 |
| **C5** | 若保留 RoPE，C1 可放宽到 ≥1 个递归块（仍建议 ≥2） | NoPE 是 attention 推到 25% 深度之后的最可能原因（**本报告推断，非官方**） |
| **C6** | 若照搬角色分离布局（M/-/* 三种独立层），M:-:* ≈ 27:25:4，总层数需 ≈ 2×（常规 block 数） | 直接读 pattern |

### 7.4 LOB 场景的两点偏离建议（本报告观点，非官方）

1. **比例可调高**。Nemotron 用 12.9% 是为 128K context 的线性成本服务；LOB 序列通常 1k–10k 步，attention 二次成本不是瓶颈。可按 Kimi Linear 的 25.9% 做上界扫描，L=12 → k ∈ {2,3} 消融。
2. **保留 NoPE 反而合适**。LOB 的「位置」是事件序号而非语言 token 位置，绝对位置意义弱；Mamba 递归 + `conv1d(k=4)` 的相对/顺序信息更贴合，且省掉 RoPE 波长与外推调参。但必须严守 C1。

### 7.5 超参映射（Nemotron → 小模型）

| 项 | Nemotron (d=4480) | 建议 |
|---|---:|---|
| `d_state` | 128 | ✅ 保持 128（与 d 无关） |
| `headdim` | 80 | ✅ 保持 64 或 80，用 n_heads 调宽 |
| `n_heads`(mamba) | 128 | `d_inner/headdim` |
| 有效 `expand` | 2.286 | ✅ 保持 ~2.0–2.3 |
| `conv_kernel` | 4 | ✅ 保持 |
| `chunk_size` | 128 | ✅ 保持（kernel 效率参数，与尺寸无关） |
| `n_groups` | 8 | `max(1, d_inner/1280)`，或降到 group_size 128/256 |
| attn `head_dim` | 128 | 小模型可降到 64 |
| attn Q:KV | 5:1 | 小模型 heads 少时直接 MHA（KV cache 非瓶颈） |
| MLP 宽度 | 3.5×d, relu², 2 矩阵 | 若改 SwiGLU 需降到 ~2.67× 才等参数量 |
| norm | RMSNorm eps=1e-5, pre-norm, 无 QK-norm | ✅ 全部照搬 |
| init | `rescale_prenorm_residual=True` | ✅ 照搬（层数变少时缩放自动变大） |
