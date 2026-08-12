#!/usr/bin/env python3
"""R7 reference 匹配的边际分析与误差注入扫描（只读）。

回答两个问题，二者都不需要等 R7 训练完：

1. **安全边际 Δ_min**：同 (side, price, original_size) 的干扰订单中，创建时间离目标
   最近的那个有多近。因为一次 touch 里所有订单共享 now_ns，
   `ref_age(target) - ref_age(o) == born_ns(o) - born_ns(target)`，所以 Δ_min 是纯静态量。
   Δ_min 直接给出「模型 ref_age 预测误差要小于多少才不会指错单」。

2. **误差注入下，截断匹配 vs 最近邻匹配**：给真值 ref_age 注入 ±δ，比较
   (a) 截断到某精度后要求唯一命中（Codex 现有审计与用户设计里「回退到 μs/ms」的做法）
   (b) 在同 (side,price,size) 内取 ref_age 最接近预测值的那一笔（最近邻）
   两者在同一 δ 下的 exact-target 命中率。

判据：如果最近邻在粗精度上并不比截断差，甚至更好，那么「ns→μs→ms 逐级回退」这段
代码就不必写——换成一次最近邻匹配即可，且不会引入截断特有的桶边界脆弱性。

数据：paired-255 的真实序列（data_cond 建可见状态 + data_real 计数），
order_id 即 ground truth，故本脚本报的是 exact-target 口径。
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

ROOT = "/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/varlen_eval255"
COND = f"{ROOT}/data_cond"
REAL = f"{ROOT}/data_real"

# 事件类型：2=partial cancel, 3=delete, 4=execution
EVENTS = ("2", "3", "4")
EVENT_NAME = {"2": "partial_cancel", "3": "delete", "4": "execution"}

QUANTA = {"sec": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}

# 注入误差幅度（ns）。0 是无误差对照；1e9 = 1 秒。
DELTAS_NS = [0, 1, 10, 100, 1_000, 10_000, 100_000,
             1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

# Δ_min 的报告阈值（ns）
MARGIN_THRESHOLDS = [1, 10, 100, 1_000, 10_000, 100_000,
                     1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

INF = float("inf")


@dataclass
class Order:
    oid: str
    side: int
    price: int
    original_size: int
    remaining_size: int
    born_ns: int


def parse_ns(text: str) -> int:
    """CSV 的十进制秒 → 整数纳秒。走 Decimal 避免浮点往返丢精度。"""
    return int(Decimal(text) * Decimal(10**9))


def rows(path: str):
    with open(path, newline="") as handle:
        yield from csv.reader(handle)


def percentiles(sorted_vals, qs):
    """已排序列表的分位数，空列表返回 None。"""
    if not sorted_vals:
        return {str(q): None for q in qs}
    out = {}
    n = len(sorted_vals)
    for q in qs:
        idx = min(n - 1, max(0, int(round(q / 100.0 * (n - 1)))))
        out[str(q)] = sorted_vals[idx]
    return out


def main() -> None:
    # --- 累加器 -------------------------------------------------------------
    total = {e: 0 for e in EVENTS}          # 全部 touch
    visible = {e: 0 for e in EVENTS}        # 目标在可见历史内
    margins = {e: [] for e in EVENTS}       # Δ_min（可见目标；无干扰记为 INF）
    no_rival = {e: 0 for e in EVENTS}       # 完全没有干扰订单

    # 命中计数：method -> precision -> delta -> event -> count
    hits = {
        method: {p: {d: {e: 0 for e in EVENTS} for d in DELTAS_NS} for p in QUANTA}
        for method in ("truncate", "nearest")
    }
    # 截断法的「唯一但选错了」计数（最近邻法没有 ambiguous 概念，单列）
    wrong = {
        method: {p: {d: {e: 0 for e in EVENTS} for d in DELTAS_NS} for p in QUANTA}
        for method in ("truncate", "nearest")
    }

    def process(path: str, active: dict, index: dict, count: bool) -> None:
        for row in rows(path):
            if len(row) < 6:
                continue
            now_ns = parse_ns(row[0])
            event_type, order_id = row[1], row[2]
            quantity, price, side = int(row[3]), int(row[4]), int(row[5])

            if event_type == "1":  # NEW
                order = Order(order_id, side, price, quantity, quantity, now_ns)
                active[order_id] = order
                index[(side, price, quantity)][order_id] = order
                continue

            if event_type not in EVENTS:
                continue

            if count:
                total[event_type] += 1

            target = active.get(order_id)

            if count and target is not None:
                visible[event_type] += 1
                key = (target.side, target.price, target.original_size)
                pool = list(index[key].values())          # 含 target 自身
                rivals = [o for o in pool if o.oid != target.oid]

                # --- 1. 安全边际 -------------------------------------------
                if rivals:
                    dmin = min(abs(o.born_ns - target.born_ns) for o in rivals)
                    margins[event_type].append(dmin)
                else:
                    no_rival[event_type] += 1
                    margins[event_type].append(INF)

                # --- 2. 误差注入 -------------------------------------------
                true_age = now_ns - target.born_ns
                # 候选订单在簿里永远是原始纳秒；精度只限制模型能输出什么
                pool_ages = [(o, now_ns - o.born_ns) for o in pool]

                for precision, quantum in QUANTA.items():
                    # 模型只能输出 `precision` 粒度 → 真值先截断到该粒度
                    quantized_true = (true_age // quantum) * quantum
                    for delta in DELTAS_NS:
                        # ±δ 各测一次再折半计入，保证确定性（不用随机数）
                        for signed in ((0,) if delta == 0 else (delta, -delta)):
                            pred_age = quantized_true + signed
                            weight = 1 if delta == 0 else 0.5

                            # (a) 截断法：pred 与候选同粒度分桶，唯一才接受。
                            #     找不到或有歧义都算失败 —— 不会选错单。
                            pred_bucket = pred_age // quantum
                            matches = [
                                o for o, age in pool_ages
                                if age // quantum == pred_bucket
                            ]
                            if len(matches) == 1:
                                bucket = hits if matches[0].oid == target.oid else wrong
                                bucket["truncate"][precision][delta][event_type] += weight

                            # (b) 最近邻法：候选保持原始纳秒，取距 pred 最近的一笔。
                            #     它几乎永不失败，代价是会静默选错（26tok L2 的病）。
                            best, best_dist = None, None
                            for o, age in pool_ages:
                                dist = abs(age - pred_age)
                                if best_dist is None or dist < best_dist:
                                    best, best_dist = o, dist
                                elif dist == best_dist:
                                    best = None  # 平局 → 歧义，判失败
                            if best is not None:
                                bucket = hits if best.oid == target.oid else wrong
                                bucket["nearest"][precision][delta][event_type] += weight

            # --- 状态推进（无论是否计数都要做）--------------------------------
            if target is None:
                continue
            if event_type in ("2", "4"):
                target.remaining_size -= quantity
                if target.remaining_size <= 0:
                    active.pop(order_id, None)
                    index[(target.side, target.price,
                           target.original_size)].pop(order_id, None)
            else:
                active.pop(order_id, None)
                index[(target.side, target.price,
                       target.original_size)].pop(order_id, None)

    message_files = sorted(glob.glob(f"{COND}/*message*.csv"))
    max_seq = int(os.environ.get("MAX_SEQ", "0"))
    if max_seq > 0:
        message_files = message_files[:max_seq]
    for condition_path in message_files:
        real_path = f"{REAL}/{os.path.basename(condition_path)}"
        active: dict = {}
        index: dict = defaultdict(dict)
        process(condition_path, active, index, count=False)
        process(real_path, active, index, count=True)

    # --- 汇总 ---------------------------------------------------------------
    total_all = sum(total.values())
    visible_all = sum(visible.values())
    all_margins = sorted(m for e in EVENTS for m in margins[e] if m != INF)
    n_inf = sum(1 for e in EVENTS for m in margins[e] if m == INF)

    margin_report = {
        "visible_targets": visible_all,
        "no_rival_count": n_inf,
        "no_rival_percent": 100.0 * n_inf / visible_all if visible_all else None,
        "finite_margin_count": len(all_margins),
        "percentiles_ns": percentiles(
            all_margins, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
        "cumulative_at_or_below_ns": {
            str(t): {
                "count": sum(1 for m in all_margins if m <= t),
                "percent_of_visible": 100.0 * sum(1 for m in all_margins if m <= t) / visible_all,
            }
            for t in MARGIN_THRESHOLDS
        },
        "by_event": {
            EVENT_NAME[e]: {
                "visible": visible[e],
                "no_rival": no_rival[e],
                "median_finite_ns": (
                    sorted(m for m in margins[e] if m != INF)[
                        len([m for m in margins[e] if m != INF]) // 2]
                    if any(m != INF for m in margins[e]) else None
                ),
            }
            for e in EVENTS
        },
    }

    injection_report = {}
    for method in ("truncate", "nearest"):
        injection_report[method] = {}
        for precision in QUANTA:
            injection_report[method][precision] = {}
            for delta in DELTAS_NS:
                h = sum(hits[method][precision][delta].values())
                w = sum(wrong[method][precision][delta].values())
                injection_report[method][precision][str(delta)] = {
                    "hit": h,
                    "wrong": w,
                    "percent_of_visible": 100.0 * h / visible_all if visible_all else None,
                    "percent_of_all_touch": 100.0 * h / total_all if total_all else None,
                    "wrong_percent_of_visible": 100.0 * w / visible_all if visible_all else None,
                }

    result = {
        "sequences": len(message_files),
        "denominators": {
            "all_touch": total_all,
            "visible_target": visible_all,
            "by_event_total": {EVENT_NAME[e]: total[e] for e in EVENTS},
            "by_event_visible": {EVENT_NAME[e]: visible[e] for e in EVENTS},
        },
        "margin": margin_report,
        "error_injection": injection_report,
        "notes": {
            "ground_truth": "order_id from data_real; therefore exact-target semantics",
            "delta_signs": "each non-zero delta measured at +d and -d, weight 0.5 each",
            "nearest_tie": "tie in distance counted as failure (ambiguous), not a coin flip",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
