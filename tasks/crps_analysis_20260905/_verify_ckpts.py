import hashlib, json
from pathlib import Path
import numpy as np, jax
import orbax.checkpoint as ocp

T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt")

def manifest(state_dir):
    # names + sizes of every file under state/ -- catches truncation and structural drift
    rows = sorted((str(p.relative_to(state_dir)), p.stat().st_size)
                  for p in state_dir.rglob("*") if p.is_file())
    h = hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:12]
    return len(rows), sum(s for _, s in rows), h, rows

ref_state = T/"wm_ft_traj3_s1"/"step_1200"/"69378"/"state"
rn, rb, rh, rrows = manifest(ref_state)
ref_names = {n for n, _ in rrows}
print(f"reference traj3_s1/step_1200: {rn} files, {rb:,} B, layout {rh}\n")
print(f"{'seed':6s}{'files':>7s}{'bytes':>15s}{'names==ref':>12s}{'max |dsize|':>13s}")
sizes = {}
for s in [40,41,42,43,44,45,46,47]:
    d = sorted(T.glob(f"wm_ft_r3rep_s{s}/*_step1200"))[0]/"69378"/"state"
    n, b, h, rows = manifest(d)
    names = {x for x, _ in rows}
    same = "yes" if names == ref_names else f"NO(+{len(names-ref_names)}/-{len(ref_names-names)})"
    rd = dict(rrows); dmax = max((abs(sz - rd.get(nm, sz)) for nm, sz in rows), default=0)
    sizes[s] = b
    print(f"s{s:<5d}{n:>7d}{b:>15,d}{same:>12s}{dmax:>13,d}")
print()
print(f"{'full restore':22s}{'arrays':>9s}{'elements':>14s}{'finite':>9s}")
for tag, d in [("r3rep_s40", sorted(T.glob("wm_ft_r3rep_s40/*_step1200"))[0]/"69378"/"state"),
               ("r3rep_s47", sorted(T.glob("wm_ft_r3rep_s47/*_step1200"))[0]/"69378"/"state"),
               ("traj3_s1 (ref)", ref_state)]:
    tree = ocp.PyTreeCheckpointer().restore(d)
    arrs = [np.asarray(x) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "dtype")]
    fl = [a for a in arrs if a.dtype.kind == "f"]
    ok = all(np.isfinite(a).all() for a in fl)
    print(f"{tag:22s}{len(arrs):>9d}{sum(a.size for a in arrs):>14,d}{('yes' if ok else 'NO'):>9s}")
