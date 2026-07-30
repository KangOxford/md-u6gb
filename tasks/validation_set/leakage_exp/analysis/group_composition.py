import numpy as np, csv
base = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set"
cum, years, months = [], [], []
with open(f"{base}/artifacts_valset_v1_j5790795/files_48mo.csv") as f:
    for row in csv.DictReader(f):
        cum.append(int(row["cum_start"])); years.append(int(row["date"][:4])); months.append(row["date"][:7])
cum = np.array(cum); years = np.array(years)
def dist(name, path):
    idx = np.load(path)
    fi = np.searchsorted(cum, idx, side="right") - 1
    y = years[fi]
    out = {int(yy): float((y == yy).mean()) for yy in (2022, 2023, 2024, 2025)}
    print(f"{name:5s} " + "  ".join(f"{yy}:{out[yy]*100:5.2f}%" for yy in sorted(out)))
    return y
for name, p in [("SEEN", f"{base}/leakage_exp/groups/group_seen.npy"),
                ("MID",  f"{base}/leakage_exp/groups/group_mid.npy"),
                ("VAL",  f"{base}/leakage_exp/groups/group_val.npy")]:
    dist(name, p)
