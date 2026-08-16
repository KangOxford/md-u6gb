#!/usr/bin/env python3
"""ledger_kit 用例：Hyper-XVLA 实验全史泳道图（2026-05 至 2026-08）。

一行一个实验身份、一段一次运行、空洞是中断——sgpur 泳道语义。
日期出处：/projects/public/u6gb/.claude/plans/compiled-questing-fairy.md 的
时间线与实验记录表（job id 全部真实）。

用法：
    python3 example_xvla.py            # 写出 example.html
    from example_xvla import ROWS      # 或在 notebook 里直接 render
"""

import os
import sys
from datetime import datetime as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ledger import Row, Seg, render_ledger  # noqa: E402

ROWS = [
    Row("init v3→v9", "5月", "p1seed 初始化演进（frozen-VLM bug 修复线）",
        segs=[Seg(D(2026, 5, 20), D(2026, 5, 24, 12), "v3 … v9_p1seed", "full",
                  tip="5 月下旬：frozen VLM copy bug 定案后，init v3 起修复，"
                      "5-24 生成 v9_p1seed（sliced seeding）")],
        tail=("7 版", "v9 存活"), mark="done"),
    Row("4853407", "4N", "hyper h1024/d6 lora 预训练 + LIBERO 8/10",
        segs=[Seg(D(2026, 5, 25), D(2026, 5, 29, 18), "80k 预训练 → 8/10", "full",
                  tip="job 4853407 链：p1seed init 起步，80k 预训练，"
                      "LIBERO spatial 8/10（2026-05-29）\n"
                      "后续速度实测的 checkpoint 血统即此")],
        tail=("1 链", "8/10 ✓"), mark="done"),
    Row("5289175", "4N", "baseline 200K 第一段",
        segs=[Seg(D(2026, 6, 18, 12), D(2026, 6, 19, 12), "0 → 45800", "full", bad=True,
                  tip="job 5289175：24h walltime，无接续链\n"
                      "停点 step 45800/200000，ckpt-40000")],
        tail=("45800 步", "TIMEOUT"), mark="TIMEOUT", mark_bad=True),
    Row("5285200", "4N", "hyper h192 直出头（发散，废线）",
        segs=[Seg(D(2026, 6, 18, 12), D(2026, 6, 19, 12), "loss 3075→4878 ↑", "low", bad=True,
                  tip="job 5285200：h192 容量/头配置下 loss 单调爆炸，永不回落\n"
                      "6-21 判定 collapse，方向废弃")],
        tail=("1 段", "✗ 发散"), mark="废弃", mark_bad=True),
    Row("5333774", "1N", "速度基准（CUDA Graph B=1）",
        segs=[Seg(D(2026, 6, 21, 9), D(2026, 6, 21, 15), "2.22 vs 9.27 ms", "full",
                  tip="job 5333774：4853407 checkpoint 实测\n"
                      "hyper 2.22±0.02 ms vs baseline 9.27±0.03 ms = 4.18×")],
        tail=("1 run", "4.18× ✓"), mark="done"),
    Row("5694855", "4N", "hyper lora run-1（未播种，停滞）",
        segs=[Seg(D(2026, 7, 17, 17), D(2026, 7, 19, 6), "loss 停滞于 9–15", "mid", bad=True,
                  tip="job 5694855：recipe 现做 fresh init，"
                      "--seed_hyper_from_xvla_transformer 未传（default=False）\n"
                      "base 零矩阵起步 × LR 5e-6 ⇒ 19400 步停滞于 9–15\n"
                      "此后停摆一个月（7-19 → 8-15），空洞即中断")],
        tail=("19400 步", "✗ 停滞"), mark="TIMEOUT", mark_bad=True),
    Row("6023162+", "4N", "hyper lora run-2（p1seed 重启）",
        segs=[Seg(D(2026, 8, 15, 17, 6), D(2026, 8, 16, 17, 5), "263→8.6@200→0.4", "full",
                  tip="job 6023162：v9_p1seed init 重启，诊断闸门 500 步内全绿\n"
                      "接续链 afterany 自动滚动")],
        tail=("~8%", "→ 200K"), mark="→ running"),
    Row("6023487+", "4N", "baseline 200K 重启链",
        segs=[Seg(D(2026, 8, 15, 17, 47), D(2026, 8, 16, 1, 37), "40000 →", "full", bad=True,
                  tip="job 6023487：ckpt-40000 续跑\n"
                      "08-16 01:37 SIGABRT@ckpt-50000 保存时刻（inode 配额打满窗口）"),
              Seg(D(2026, 8, 16, 2, 40), D(2026, 8, 16, 17), "→ 200K", "full",
                  tip="job 6026878：链自动重启，安然越过原 crash 位")],
        tail=("~30%", "→ 200K"), mark="→ running"),
    Row("6023209+", "4N", "vanilla h192 消融（首次全量）",
        segs=[Seg(D(2026, 8, 15, 17, 37), D(2026, 8, 16, 1, 37), "0 →", "full", bad=True,
                  tip="job 6023209：direct-head 消融臂首次全量\n08-16 凌晨 SIGTERM@8h"),
              Seg(D(2026, 8, 16, 2, 40), D(2026, 8, 16, 17), "→ 200K", "full",
                  tip="job 6026879：链自动重启")],
        tail=("~4%", "→ 200K"), mark="→ running"),
    Row("libero-eval", "1N", "LIBERO 10×50 判决评测（现货 ckpt 对）",
        segs=[Seg(D(2026, 8, 16, 2, 0), D(2026, 8, 16, 6, 10), "70% vs 100%", "full",
                  tip="attach + sbatch 兜底 + 独占仲裁三测\n"
                      "hyper 70.0% vs baseline 100.0%（30.0pp）× 4.16× 速度")],
        tail=("3 测", "判决 ✓"), mark="done"),
]

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.html")
    html = render_ledger(
        "Hyper-XVLA 实验全史",
        "one row per experiment · blocks are runs · the holes are the breaks · 2026-05 → 2026-08",
        ROWS, now=D(2026, 8, 16, 11, 30))
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({len(html)} bytes)")
