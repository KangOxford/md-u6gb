"""Re-examine three claims I made too strongly. Reuses measure_real's loaders."""
import sys, json, itertools, math, re
sys.path.insert(0, "/home/u6gb/kangli.u6gb/crps_runspread_20260905")
import numpy as np
from measure_real import load_cells, build, day_boot, ci, HARV

daymap = json.load(open(HARV/"daymap_META.json"))
runs, real = load_cells(r"wm_ft_traj_s\d+")
ids, y, days, X, labels = build(runs, real, daymap)   # X: (n_ctx, R, S)
nctx, R, S = X.shape
print(f"R = {R} runs, S = {S} generation members, n_ctx = {nctx}, n_days = {len(np.unique(days))}\n")

# ---- per-context estimators -------------------------------------------------
W_ctx = np.abs(X[:, :, 0] - X[:, :, 1]).mean(1)                       # within run
pairs = list(itertools.combinations(range(R), 2))
D_same, D_diff = [], []
for a, b in pairs:
    D_same.append(np.abs(X[:, a, 0] - X[:, b, 0]))
    D_same.append(np.abs(X[:, a, 1] - X[:, b, 1]))
    D_diff.append(np.abs(X[:, a, 0] - X[:, b, 1]))
    D_diff.append(np.abs(X[:, a, 1] - X[:, b, 0]))
D_same = np.mean(D_same, 0); D_diff = np.mean(D_diff, 0)
D_all  = (D_same + D_diff) / 2
W, Dd, Da = W_ctx.mean(), D_diff.mean(), D_all.mean()

print("=== correction 1: is Dbar < Wbar beyond Monte Carlo error? ===")
for tag, Dv, arr in [("decoupled (different gen seeds)", Dd, D_diff),
                     ("coupled   (all cross pairs)    ", Da, D_all),
                     ("same-seed cross pairs only     ", D_same.mean(), D_same)]:
    g = arr - W_ctx
    c = ci(day_boot(days, lambda s_: g[s_].mean()))
    print(f"  {tag}  Dbar-Wbar = {g.mean()*1e5:+.4f}e-5  95% CI [{c[0]*1e5:+.4f},{c[1]*1e5:+.4f}]  "
          f"{'excludes 0' if c[0]*c[1] > 0 else 'includes 0'}")
print("\n  The population inequality 2E|Xi-Xj| >= E|Xi-Xi'| + E|Xj-Xj'| holds in EXPECTATION.")
print("  A finite-sample estimate can fall below it by chance, so the observed deficit is")
print("  evidence of coupling only if it clears its own error bar. How often would independent")
print("  draws with this design produce it?")
rng = np.random.default_rng(20260905)
sd_ctx = X.reshape(nctx, -1).std(1, ddof=1)
hits = 0; NSIM = 4000
for _ in range(NSIM):
    Z = rng.normal(0, 1, size=(nctx, R, S)) * sd_ctx[:, None, None]
    w = np.abs(Z[:, :, 0] - Z[:, :, 1]).mean(1)
    ds = np.mean([np.abs(Z[:, a, k] - Z[:, b, k]) for a, b in pairs for k in (0, 1)], 0)
    dd = np.mean([np.abs(Z[:, a, k] - Z[:, b, 1 - k]) for a, b in pairs for k in (0, 1)], 0)
    if ((ds + dd) / 2).mean() < w.mean():
        hits += 1
print(f"  independent-draw simulation, same shapes: Dbar_all < Wbar in {hits}/{NSIM} = {hits/NSIM:.4%}")
print(f"  observed coupled deficit {(Da-W)*1e5:+.4f}e-5 against a decoupled CI half-width of "
      f"{(ci(day_boot(days, lambda s_: (D_diff-W_ctx)[s_].mean()))[1]-Dd+W)*1e5:.4f}e-5")

print("\n=== correction 2: three distinct dependencies, kept apart ===")
m0, m1 = X[:, :, 0], X[:, :, 1]
print(f"  (a) WITHIN one ensemble: the K=2 members are two generation seeds of ONE checkpoint.")
print(f"      member means across runs: {m0.mean():+.6f} vs {m1.mean():+.6f};  "
      f"sds {m0.std(ddof=1):.6f} vs {m1.std(ddof=1):.6f}")
print(f"      exchangeable given the checkpoint, so fair CRPS is unbiased FOR THAT RUN. Sharing")
print(f"      seeds across runs does not touch this.")
r_same = np.mean([np.corrcoef(X[:, a, 0], X[:, b, 0])[0, 1] for a, b in pairs])
r_diff = np.mean([np.corrcoef(X[:, a, 0], X[:, b, 1])[0, 1] for a, b in pairs])
print(f"  (b) ACROSS runs: common random numbers. cross-run r  same seed {r_same:+.4f}  "
      f"different seed {r_diff:+.4f}")
# what CRN does to a PAIRED difference
per_run_same = np.array([[np.abs(X[:, a, k] - y).mean() for k in (0, 1)] for a in range(R)])
d_paired_same, d_paired_cross = [], []
for a, b in pairs:
    d_paired_same.append(np.abs(X[:, a, 0]-y).mean() - np.abs(X[:, b, 0]-y).mean())
    d_paired_cross.append(np.abs(X[:, a, 0]-y).mean() - np.abs(X[:, b, 1]-y).mean())
print(f"      sd of a run-to-run difference, matched seed {np.std(d_paired_same, ddof=1)*1e5:.4f}e-5 "
      f"vs crossed seed {np.std(d_paired_cross, ddof=1)*1e5:.4f}e-5")
print(f"      -> for PAIRED comparisons CRN is variance reduction, not a defect. It biases only")
print(f"         estimators that assume cross-run independence, e.g. a pooled Dbar.")
print(f"  (c) EVALUATION Monte Carlo error: {nctx} contexts x {S} members, day-clustered.")
c = ci(day_boot(days, lambda s_: W_ctx[s_].mean()))
print(f"      Wbar {W*1e5:.4f}e-5, 95% CI [{c[0]*1e5:.4f},{c[1]*1e5:.4f}] -> +/-{(c[1]-c[0])/2/W*100:.1f}%")

print("\n=== correction 3: what the K=2 data can and cannot say about K ===")
print("  Algebra (verifiable by simulation, NOT by these data):")
print("    E[fair_K] - CRPS(mixture)        = (Wbar - Dbar) / (2K)   -> shrinks with K")
print("    E[fair_K] - mean single-run CRPS = (Wbar - Dbar) / 2      -> no K in it")
print("  The second is a statement about which estimand you are aiming at, not a measurement.")
sp = json.load(open(HARV/"measured_spreads.json"))
print(f"  Its MAGNITUDE is estimated from K=2 data only: {sp['bias_vs_single_run_pct']:+.3f}% "
      f"CI [{sp['bias_vs_single_run_pct_ci'][0]:+.3f},{sp['bias_vs_single_run_pct_ci'][1]:+.3f}] "
      f"-- the CI includes 0.")
print("  With every record at K=2 there is NO K ladder, so no K-dependence of anything is")
print("  measurable here. 'Flat in K' is the algebra; it must not be reported as a measurement.")
