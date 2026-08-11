# R6/R7 reference 时间精度只读审计

时间：2026-08-11T19:17:38Z  
状态：只读测试完成，等待用户选择；未创建 R7 worktree，未修改模型或提交训练。

## 固定口径

- 数据：paired-255；每条序列先用 250 条 condition 建立可见订单状态，再审计 250 条真实 continuation。
- touch：CANCEL、DELETE、EXECUTION，共 31,630 条。
- 可见目标：27,789 条；其余 3,841 条目标早于 condition 窗口。
- 内容键：`side + price + original_ref_size + t_ref`。
- 时间计算：直接把 CSV 十进制时间转换成整数纳秒，再用整数除法截断到秒、毫秒或微秒；没有浮点往返。

## `dt_ns` 与 `t_ref` 不是同一个字段

- R6/R6.1 的 `dt_ns` 没有被降精度：它直接编码当前事件相对上一事件的完整纳秒差，累加后无损重建当前事件时钟。
- 本审计里的 `t_ref` 是被 touch 的旧 NEW 订单的创建时间，或等价的 reference age。秒/毫秒/微秒/纳秒扫描只是在测第二重内容键需要多细的 reference 时间，不是在改变 `dt_ns`。
- Resolver 可以在 NEW 到来时从累计 `dt_ns` 自动保存其精确创建时间，不需要每条消息重复携带一份完整绝对时钟。只有当 L2 要靠内容键独立选中旧订单时，模型才需要给出 `t_ref`/reference age 的某种精度，或者由 L1 `ref_code` 指认。

## 结果

| `t_ref` 精度 | 可见目标内唯一数 | 可见目标内唯一率 | 全部 touch 覆盖率 |
|---|---:|---:|---:|
| 秒 | 24,608 / 27,789 | 88.5530% | 77.7996% |
| **毫秒** | **26,222 / 27,789** | **94.3611%** | **82.9023%** |
| **微秒** | **27,239 / 27,789** | **98.0208%** | **86.1176%** |
| 纳秒 | 27,789 / 27,789 | 100.0000% | 87.8565% |

remaining `ref_size` 在毫秒口径为 94.4079%，只比 original size 高 0.0468 个百分点；微秒口径两者完全相同，都是 98.0208%。因此时间精度选择不改变“优先使用静态 original `ref_size`”的判断。

这些数字是数据侧 oracle 唯一识别率，不是模型生成后的 reference 成功率。选定 E-ms 或 E-us 后，真实 L2 增量仍须通过小规模训练和 paired-255 重放测量。

## 可复现产物

- 脚本：`tasks/varlen_bench_subset_20260809/resolver_design_supervision/r6p1_twolevel_options_20260811T185757Z/audit_time_precision.py`
- 结果：`tasks/varlen_bench_subset_20260809/resolver_design_supervision/r6p1_twolevel_options_20260811T185757Z/time_precision_results.json`
- 脚本 SHA256：`dee195cc604288bdb5260a9c423c47149bb38a780ac5f8d65e7a8479b0466d25`
- 结果 SHA256：`ae42b3e6c35e88a5b30c03f7e4270db5f0551f3c53cc209dad6bd665a4e273bb`
- Notion：<https://app.notion.com/p/3b812c4568fd8145a418c3dc0a0f42d6>
