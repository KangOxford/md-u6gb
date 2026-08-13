# Mid-training 管线图（mermaid 版）

保存即在 Recent Markdown 置顶；右键 → Open Preview 渲染。与 `fig_midtrain_pipeline.drawio` 同构。

## 主循环：On-policy 终点密度比蒸馏

```mermaid
flowchart LR
    subgraph GEN[生成（前向经历马尔可夫链）]
        A[真实 LOB 流<br/>LOBSTER, event time] --> C[Context C<br/>250 条真实消息]
        C --> M[预训练 AR 模型 πθ<br/>本轮冻结]
        M -- 采样 K 条 --> G[250 条生成消息<br/>= action 序列]
        G --> E[撮合引擎 g<br/>确定性, 不可微]
        E --> R["终点收益率<br/>r = log(m500/m250)"]
    end

    subgraph W[重加权（逐票）]
        R --> KDE["密度比权重<br/>w(r) = p̂_real(r) / p̂_θ(r)<br/>log-KDE · clip ±2.5 · 自归一 · ESS 审计"]
    end

    subgraph TRAIN[训练（sparse terminal reward）]
        KDE --> MLE["加权 MLE（全参数）<br/>max Σ wᵢ·log πθ(序列ᵢ)<br/>≡ REINFORCE, reward = w"]
        C -. 真实延续 .-> ANC[real-CE 锚<br/>交替步, λ=1]
        ANC --> MLE
        MLE --> UPD[参数更新 θ → θ′]
    end

    UPD -. "IPF 下一轮：重采样、重估 p̂θ′<br/>停机：sd 增量 &lt; 0.02 或安全闸门" .-> M

    subgraph EVAL[评估]
        UPD --> P[Primary: fair CRPS · qL1 · sd · 双尾<br/>天块 bootstrap]
        UPD --> S[Safety: held-out CE · LOB-Bench 21 特征]
    end
```

## 多票扩量版（当前在跑的 wm_ft_multi）

```mermaid
flowchart TB
    subgraph DUMP[素材扇出 · 16 卡]
        T1[8 票 × 600 契机 × 6 seeds<br/>= 48 个 dump run]
    end
    subgraph WTS[逐票权重]
        T1 --> W1[每票各自 KDE 估 w · 各自自归一<br/>S9 教训: GOOG 的 w 会把 MSFT 推反]
        W1 --> MERGE[合并 28,800 条权重<br/>root 名消歧]
    end
    subgraph FT[混池训练 · 单卡]
        MERGE --> TR[wm_ft_multi 全参数<br/>2400 批/epoch · lr 1e-5 · λ=1<br/>hold 探针跨池等距 16 批]
    end
    TR --> EV[八票同索引配对评估<br/>real vs base vs multi<br/>矩阵图 + 记分板 + LOB-Bench]
```

产物对照：`fig_midtrain_pipeline.{drawio,pdf}`（Overleaf figures/）。
