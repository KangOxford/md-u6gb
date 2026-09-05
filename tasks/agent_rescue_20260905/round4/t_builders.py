#!/usr/bin/env python3
"""Acceptance test for the notebook-builder fixes.  Takes a directory so the SAME test can
be run against the pre-fix backups and against the shipping files: a test that only ever
sees the fixed tree cannot show that it detects the defect.

Usage:  t_builders.py <dir> [--bak <suffix>]
"""
import json, re, sys, os

SRC = "/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22"
FN = json.load(open(f"{SRC}/fix_nulls.json"))
CAL = FN["calibration_of_each_test_under_a_true_null"]
EX = FN["null_ladder"]["rung3_disjoint_seeds_same_epoch"]["exact_randomisation"]

d = sys.argv[1]
suf = ""
if "--bak" in sys.argv:
    suf = sys.argv[sys.argv.index("--bak") + 1]

def read(name):
    p = os.path.join(d, name + suf)
    return open(p, encoding="utf-8").read()

npass = nfail = 0
def check(name, ok, detail=""):
    global npass, nfail
    if ok: print(f"  PASS  {name}"); npass += 1
    else:  print(f"  FAIL  {name}\n        {detail}"); nfail += 1

# Ground truth, read from the data rather than restated.
day_t    = CAL["naive_day_level_t"]["effect_value"]        # -7.8362
tick_t   = CAL["naive_ticker_level_t"]["effect_value"]     # -5.2451
tick_max = CAL["naive_ticker_level_t"]["null_max_abs"]     # 6.8705
day_max  = CAL["naive_day_level_t"]["null_max_abs"]        # 2.2649
cross_df = FN["task7_honest_comparison"]["_crossed_model_verdict"]["effect_crossed_df"]
print(f"ground truth: day-level t {day_t:.4f} (null max {day_max:.2f}), "
      f"ticker-level t {tick_t:.4f} (null max {tick_max:.2f}), crossed df {cross_df:.2f}")
assert abs(day_t + 7.836) < 0.01 and abs(tick_t + 5.245) < 0.01

ft = read("friday_talk.py")

# B1 -- the figure must not label the tests by "what we used before/now", because that is
# what let the headline's t be attached to the ticker-level row.
# Scoped to the label literals themselves.  A global substring search also matched the
# comment that explains the fix, i.e. it measured the wrong thing: the check is about what
# the figure prints, not about whether the phrase occurs anywhere in the file.
labels = re.findall(r"\(\s*'([a-z_]+)'\s*,\s*'([^']*)'\s*\)", ft.split("tests = [")[1][:600])
check("B1 f2() labels the tests by statistic, not by before/now",
      bool(labels) and not any(("used before" in lab) or ("use now" in lab) for _, lab in labels),
      f"labels printed by the figure: {[lab for _, lab in labels]}")

# B2 -- no sentence may attribute the headline t to the ticker-level test.
bad = re.search(r"Test A,\s*\n?the one behind our original t = −7\.8", ft)
check("B2 the headline t is not attributed to the ticker-level test",
      bad is None, "found: 'Test A, the one behind our original t = -7.8 claim'")

# B3 -- the text must say where the headline actually came from.
check("B3 the day-level test is named as the source of the headline",
      re.search(r"headline t = −7\.8 came from the day-level test", ft) is not None,
      "the corrected attribution sentence is absent")

# B4 -- and must say that this test is the conservative one, with its measured null spread.
check("B4 the day-level test is described as conservative, with its null max",
      ("conservative" in ft and "2.27" in ft),
      "the conservative reading and its null max 2.27 are not both present")

nl = read("make_null_ladder_notebook.py")

# B5 -- 53.65 may not appear without its degrees of freedom on the same row.
row = [l for l in nl.splitlines() if "53.65" in l]
check("B5 every row quoting |t| = 53.65 carries its df",
      bool(row) and all(("df" in l) for l in row),
      f"rows without df: {[l.strip()[:90] for l in row if 'df' not in l]}")

# B6 -- the df quoted must be the one in the data, not a round number someone liked.
check("B6 the df quoted matches fix_nulls.json (3.24)",
      any(f"{cross_df:.2f}" in l for l in row),
      f"expected {cross_df:.2f} on the 53.65 row")

print(f"\n{npass} passed, {nfail} failed")
sys.exit(nfail)
