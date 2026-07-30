import numpy as np, csv
base = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set"
cum, years = [], []
with open(f"{base}/artifacts_valset_v1_j5790795/files_48mo.csv") as f:
    for row in csv.DictReader(f):
        cum.append(int(row["cum_start"])); years.append(int(row["date"][:4]))
cum = np.array(cum); years = np.array(years)
BSZ = 6
rng = np.random.default_rng(20260730)

def per_batch_year(idx):
    fi = np.searchsorted(cum, idx, side="right") - 1
    y = years[fi].reshape(-1, BSZ)
    maj = np.array([np.bincount(r, minlength=2026).argmax() for r in y])
    purity = (y == maj[:, None]).mean()
    return maj, purity

groups = {}
for g in ("seen", "mid", "val"):
    idx = np.load(f"{base}/leakage_exp/groups/group_{g}.npy")
    bl = np.load(f"{base}/leakage_exp/results/leak_350M-s5_{g}_batchloss.npy")
    assert len(idx) == len(bl) * BSZ, (g, len(idx), len(bl))
    maj, purity = per_batch_year(idx)
    groups[g] = (maj, bl)
    print(f"[{g}] batches={len(bl)} sorted={bool(np.all(np.diff(idx)>0))} batch-year-purity={purity*100:.2f}%")

print("\nPer-year mean CE (batch-level, majority year):")
print(f"{'year':>6} {'MID':>10} {'VAL':>10} {'VAL-MID':>10} {'SEEN':>10}")
for yy in (2022, 2023, 2024, 2025):
    row = {}
    for g in ("mid", "val", "seen"):
        maj, bl = groups[g]
        row[g] = bl[maj == yy].mean() if (maj == yy).any() else np.nan
    print(f"{yy:>6} {row['mid']:>10.6f} {row['val']:>10.6f} {row['val']-row['mid']:>10.6f} {row['seen']:>10.6f}")

# composition-adjusted VAL-MID: reweight VAL per-year CE by MID year mix, bootstrap over batches
maj_m, bl_m = groups["mid"]; maj_v, bl_v = groups["val"]
w = {yy: (maj_m == yy).mean() for yy in (2022, 2023, 2024, 2025)}
def adj_diff(bm, bv, mm, mv):
    tot = 0.0
    for yy, wy in w.items():
        sv = bv[mv == yy]; sm = bm[mm == yy]
        if len(sv) == 0 or len(sm) == 0: continue
        tot += wy * (sv.mean() - sm.mean())
    return tot
point = adj_diff(bl_m, bl_v, maj_m, maj_v)
bs = []
for _ in range(20000):
    im = rng.integers(0, len(bl_m), len(bl_m)); iv = rng.integers(0, len(bl_v), len(bl_v))
    bs.append(adj_diff(bl_m[im], bl_v[iv], maj_m[im], maj_v[iv]))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nComposition-adjusted VAL-MID (MID year weights): {point:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]")
