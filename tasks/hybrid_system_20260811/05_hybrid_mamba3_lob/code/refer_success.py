"""(7.4) refer order success rate, split by event class and by reference age.

Reads the per-message provenance bit field the generator already writes next to
each generated sequence, so this measures the resolution the run actually
performed rather than re-deriving one after the fact.

Two rates, and the difference between them is the whole point:

  L1 exact   the model named the referenced order's nanosecond creation time
             and a live order carried exactly that time. This is the ability
             the hybrid is supposed to improve.
  post-fallback  L1, plus the same-price nearest-millisecond fallback that
             fires when L1 misses. It has no distance threshold, so it never
             fails and can silently cancel a different order. High numbers
             here are a statement about the resolver, not about the model.

Usage:
  python refer_success.py --gen-dir <data_gen> [--out <dir>] [--label NAME]
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

REPO = "/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811"
sys.path.insert(0, os.path.join(REPO, "src"))

# Bit layout is owned by the generator; import it rather than restating it, so a
# change there cannot silently mean something else here.
from post_training.heuristic_learning.fidelity import (  # noqa: E402
    RESOLUTION, PROV_NAN_FIRED, PROV_PRICE_DIFFERS_FROM_REF,
)

EVENT_NAMES = {1: "new", 2: "cancel", 3: "delete", 4: "execution",
               5: "hidden", 6: "cross", 7: "halt"}
REFERENCING = (2, 3, 4)          # what the task calls cancel / delete / execution
# 默认分桶配到 250，因为 500 条窗口下 cond 是 250，再远的目标一律在窗口之外。
# 2,000 条窗口把可达距离推到 2,000，那时 251-2000 会全部塌进 ">250" 一个桶里，
# 而机制判据 P2 问的恰恰是「命中率怎么随距离变化」——塌桶等于把要测的量测没了。
# 所以分桶可配，且扩展集的前五个与默认完全相同，跨上下文可以逐桶对齐着比。
AGE_BINS_500 = [(1, 10), (11, 25), (26, 50), (51, 100), (101, 250)]
AGE_BINS_2K = AGE_BINS_500 + [(251, 500), (501, 1000), (1001, 2000)]
AGE_BINS = AGE_BINS_500


def sequence_pairs(gen_dir):
    """(provenance path, message path) for every sequence, matched by id."""
    prov, msgs = {}, {}
    for name in os.listdir(gen_dir):
        if not name.endswith(".csv"):
            continue
        if "_provenance_real_id_" in name:
            prov[name.split("_provenance_real_id_")[1]] = name
        elif "_message_real_id_" in name:
            msgs[name.split("_message_real_id_")[1]] = name
    keys = sorted(set(prov) & set(msgs))
    return [(os.path.join(gen_dir, prov[k]), os.path.join(gen_dir, msgs[k])) for k in keys]


def read_ints(path):
    with open(path) as fh:
        return [int(float(line)) for line in fh if line.strip()]


def read_messages(path):
    """LOBSTER six columns: time, event_type, order_id, size, price, direction."""
    out = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            out.append((float(row[0]), int(float(row[1])), int(float(row[2])),
                        int(float(row[3])), int(float(row[4])), int(float(row[5]))))
    return out


def age_bin(age):
    if age is None:
        return "before-window"
    for lo, hi in AGE_BINS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return f">{AGE_BINS[-1][1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--age-bins", choices=("500", "2k"), default="500",
                    help="500 keeps the published bins so old runs reproduce; "
                         "2k extends them to 2000 for long-context benches")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="arm")
    a = ap.parse_args()

    global AGE_BINS
    AGE_BINS = AGE_BINS_2K if a.age_bins == "2k" else AGE_BINS_500
    print(f"[refer] age bins: {a.age_bins} -> {AGE_BINS}")

    pairs = sequence_pairs(a.gen_dir)
    if not pairs:
        sys.exit(f"no matched provenance/message pairs under {a.gen_dir}")

    by_event = defaultdict(Counter)          # event_type -> resolution counter
    by_age = defaultdict(Counter)            # age bin -> resolution counter
    nan_fired = price_differs = 0
    n_msgs = n_seqs = 0
    length_mismatch = 0

    for prov_path, msg_path in pairs:
        codes = read_ints(prov_path)
        msgs = read_messages(msg_path)
        if len(codes) != len(msgs):
            length_mismatch += 1
            n = min(len(codes), len(msgs))
            codes, msgs = codes[:n], msgs[:n]
        n_seqs += 1
        n_msgs += len(msgs)

        # Where each order id was first submitted inside this generated window.
        # A reference to an id never submitted here points before the window,
        # which no amount of recall can reach and which is scored separately.
        first_seen = {}
        for idx, (_, ev, oid, _, _, _) in enumerate(msgs):
            if ev == 1 and oid not in first_seen:
                first_seen[oid] = idx

        for idx, (code, (_, ev, oid, _, _, _)) in enumerate(zip(codes, msgs)):
            res = RESOLUTION[code & 0b11]
            nan_fired += bool(code & PROV_NAN_FIRED)
            price_differs += bool(code & PROV_PRICE_DIFFERS_FROM_REF)
            by_event[ev][res] += 1
            if ev in (2, 3):
                src = first_seen.get(oid)
                by_age[age_bin(None if src is None else idx - src)][res] += 1

    def rates(counter):
        tot = sum(counter.values())
        if not tot:
            return None
        return {k: counter.get(k, 0) for k in RESOLUTION.values()} | {"total": tot}

    print(f"label: {a.label}")
    print(f"sequences: {n_seqs}   generated messages: {n_msgs}")
    if length_mismatch:
        print(f"WARNING: {length_mismatch} sequences had provenance/message length mismatch")
    print()

    print("resolution by event class")
    hdr = f"{'event':>12} {'n':>8} {'L1 exact':>12} {'L2 fallback':>13} {'miss':>9} {'not-a-cancel':>14}"
    print(hdr)
    print("-" * len(hdr))
    per_event = {}
    for ev in sorted(by_event):
        c = by_event[ev]
        tot = sum(c.values())
        l1, l2, miss, na = (c.get("exact-timestamp-hit", 0), c.get("price-fallback", 0),
                            c.get("total-miss", 0), c.get("not-a-cancel", 0))
        name = EVENT_NAMES.get(ev, str(ev))
        pct = lambda v: f"{100.0 * v / tot:.2f}%" if tot else "-"
        print(f"{name:>12} {tot:>8,} {pct(l1):>12} {pct(l2):>13} {pct(miss):>9} {pct(na):>14}")
        per_event[name] = {"n": tot, "exact": l1, "fallback": l2, "miss": miss,
                           "not_a_cancel": na}

    cd = Counter()
    for ev in (2, 3):
        cd.update(by_event.get(ev, {}))
    tot_cd = sum(cd.values())
    l1 = cd.get("exact-timestamp-hit", 0)
    l2 = cd.get("price-fallback", 0)
    miss = cd.get("total-miss", 0)
    print()
    print("cancel + delete combined  (the published comparison basis)")
    if tot_cd:
        print(f"  L1 exact          {l1:>8,} / {tot_cd:,} = {100.0*l1/tot_cd:.4f}%   <- primary criterion")
        print(f"  + L2 fallback     {l2:>8,}            = +{100.0*l2/tot_cd:.4f} pp")
        print(f"  = any live id     {l1+l2:>8,} / {tot_cd:,} = {100.0*(l1+l2)/tot_cd:.4f}%")
        print(f"  unresolved        {miss:>8,} / {tot_cd:,} = {100.0*miss/tot_cd:.4f}%")

    print()
    print("execution class")
    ex = by_event.get(4, Counter())
    tot_ex = sum(ex.values())
    if tot_ex and ex.get("not-a-cancel", 0) == tot_ex:
        print(f"  {tot_ex:,} execution messages, all recorded as 'not-a-cancel'.")
        print("  The generator resolves references only for event types 2 and 3;")
        print("  executions receive a fresh order id, so the model's predicted")
        print("  reference triple is discarded. An execution success rate is not")
        print("  undefined by accident here, it is undefined by construction.")
    elif tot_ex:
        print(f"  {tot_ex:,} execution messages, mixed resolutions: {dict(ex)}")
    else:
        print("  no execution messages generated")

    print()
    print("L1 exact rate by age of the referenced order  (mechanism criterion)")
    hdr2 = f"{'age (messages back)':>22} {'n':>9} {'L1 exact':>10} {'L2':>9} {'miss':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    order = [f"{lo}-{hi}" for lo, hi in AGE_BINS] + [f">{AGE_BINS[-1][1]}", "before-window"]
    per_age = {}
    for key in order:
        c = by_age.get(key)
        if not c:
            continue
        tot = sum(c.values())
        e, f, m = (c.get("exact-timestamp-hit", 0), c.get("price-fallback", 0),
                   c.get("total-miss", 0))
        print(f"{key:>22} {tot:>9,} {100.0*e/tot:>9.2f}% {100.0*f/tot:>8.2f}% {100.0*m/tot:>8.2f}%")
        per_age[key] = {"n": tot, "exact": e, "fallback": f, "miss": m}

    print()
    print(f"nan branch fired: {nan_fired:,} / {n_msgs:,} = {100.0*nan_fired/max(n_msgs,1):.4f}%")
    print(f"price differs from ref: {price_differs:,} ({100.0*price_differs/max(n_msgs,1):.2f}%)"
          "  <- not a defect count; real messages trip it at the same rate")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"refer_success_{a.label}.json")
        with open(path, "w") as fh:
            json.dump({"label": a.label, "gen_dir": a.gen_dir,
                       "sequences": n_seqs, "messages": n_msgs,
                       "by_event": per_event, "by_age": per_age,
                       "cancel_delete": {"n": tot_cd, "exact": l1,
                                         "fallback": l2, "miss": miss},
                       "nan_branch_fired": nan_fired,
                       "price_differs_from_ref": price_differs}, fh, indent=1)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
