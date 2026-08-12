# 进度

## 时间线

| 时间 (UTC) | 事件 |
|---|---|
| 11:0x | 确认 MiniMax H3 是**全模态视频+音频生成模型**，非 LLM；2026-07-31 发布 |
| 11:1x | 确认**技术报告未发布**、**训练数据未开源**——用户的两条前提均需修正 |
| 11:2x | 拉配置与 diffusers 官方移植源码，反推出完整架构 |
| 11:3x | 建 venv（diffusers 0.40.0.dev0 + transformers 4.57.6 + PyAV 18.0.0） |
| 11:42 | 提交 job **5998320**（取资产 + 官方 checkpoint 冒烟） |
| 11:5x | `h3nano.py` CPU 自检通过：**101.652 M** 参数，前向/损失/逆变换全对 |
| 12:0x | checkpoint 保存/剪枝/面包屑/resume 单独验证通过 |
| 12:1x | 采样回路（双调度器）CPU 验证通过 |
| 12:2x | 提交 job **5998498**（阶段 2–6 全流水线，依赖 5998320） |

## 作业

| Job ID | 名称 | 状态 | 说明 |
|---|---|---|---|
| 5998320 | h3-fetch-smoke | PENDING | 1 节点 6h。下载 143 GB checkpoint + 34 GB VGGSound，参数普查，T2VA 冒烟 |
| 5998498 | h3-nano-pipeline | PENDING (Dependency) | 1 节点 12h。语料 → 闸门 → 预训练 → FL2VA → CFG 蒸馏 → 评估 |

排队原因：账户 fair-share 已用 82%，`FairShare=0.038`，全部作业 `Priority=1`。无 QOS/账户 TRES 硬限额，集群此刻有 99 节点空闲，等 backfill。

## 已完成并验证的部分

| 组件 | 验证方式 | 结果 |
|---|---|---|
| `h3nano.py` 模型 | CPU 实例化 + 前向 | 101.652 M；AdaLN 35.4% / FFN 36.6% / attn 24.4%（33B 对应 39.4/35.0/23.3） |
| packed 布局 | 调用参考实现 `build_packed_sequence` | seq=1748（video 1408 + text 96 + audio 244） |
| 几何公式 `17n+5 → 5n+2` | 源码交叉验证 + 旋转时钟自洽 | F=73 → 22 latent 帧；4×17+(1+4)=73 ✓ |
| patchify / audio pack | 往返逆变换 | max\|err\| = 0.00e+00（两者） |
| 流匹配目标 | 从 scheduler 代数推导 | `v = x0 − ε` |
| 训练循环 | CPU 干跑 | 前向/反向/优化器/日志全通（登录节点 4 GiB cgroup 限制导致长跑被杀，非代码问题） |
| checkpoint | 独立测试 | 原子面包屑 ✓、精确保留 3 个 ✓、resume 状态一致 ✓ |
| 采样回路 | CPU，5 步 | guidance=0 → 4 次前向；guidance=5 → 8 次前向；输出有限 ✓ |
| 双 shift | scheduler σ 网格 | video 5 步中 4 步在 σ>0.75；audio 均匀分布 |

## 待办

- [ ] 5998320 落地：验证「13B 在 AdaLN」的实测值、packed 布局实测、官方 T2VA 出片
- [ ] 5998498 落地：E1/E3/E4/E6/E7 五项指标
- [ ] E5（双 shift 消融）需 3 组独立训练，视预算决定
- [ ] E2（蒸馏前后前向次数与墙钟）在流水线阶段 6 内测

## 遇到的问题与处理

| 问题 | 根因 | 处理 |
|---|---|---|
| `hf download --exclude` 把 json 全滤掉 | glob 语义与预期不符 | 改用 curl 点名拉 19 个配置文件 |
| diffusers 0.37.1 无 MiniMaxH3 | 只在 main 分支 | venv 装 git main（0.40.0.dev0） |
| `cannot import HybridCache` | diffusers main 要 transformers 4.x，宿主是 5.5.0 | venv 内钉 `transformers>=4.57,<5` |
| `libgomp: Thread creation failed` | 登录节点 pids.max | `taskset -c 0-3` + `OMP_NUM_THREADS=4` |
| 训练干跑 exit 137 | 登录节点 cgroup 上限 **4 GiB**，已用 3.31 GB | 确认是登录节点限制；计算节点 856 GB 无此问题 |
| 参数普查 `token_refiner` 只有 0.003 M | 分桶时 `.attn.` 判断在 `token_refiner` 之前，抢走了 refiner 的 attn/ffn | 调整判断顺序（两个文件都改） |
