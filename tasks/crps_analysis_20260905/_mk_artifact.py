#!/usr/bin/env python
"""Emit the audit as a designed, self-contained HTML page (figures as data URIs)."""
import base64, json, html
from pathlib import Path

O = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905")
FIG = O / "figs_audit"

def img(name, alt):
    b = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{html.escape(alt)}">'

LEDGER = [
 ("Panel effect, fair CRPS",           "+2.92%, p = 0.094",   "+2.92%, p = 0.594",          "Run-to-run spread was never in the standard error", True),
 ("Peak refutation, multi4",           "P = 0.19",            "P = 0.0970",                 "Max-of-normals replaced by exact Grubbs", True),
 ("Peak refutation, unifw (control)",  "P = 0.1385",          "P = 0.0053",                 "Same, and it now fires in the control arm", True),
 ("Null band, 2-seed vs 4-seed",       "0.0195",              "0.0238",                     "Band was built for a 4-vs-4 contrast", False),
 ("Variance ladder, rung 3 / rung 2",  "2.89&times;",         "4.08&times;",                "A level sd was compared against a contrast sd", False),
 ("Welch df from n_eff = 2.365",       "1.37",                "2.30",                       "n_eff &minus; 1 is not the Satterthwaite reduction", False),
 ("maxT family-wise error rate",       "0.042 claimed",       "0.076&ndash;0.082 measured",  "Calibrated on the single-contrast curve", False),
 ("Cosine LR confound at step 1200",   "8.9&times;",          "1.0&times; &mdash; no schedule exists", "A metadata field that reaches nothing", True),
 ("Sign-flip p, all three contrasts",  "0.0078",              "0.0078 = the floor",         "Not a measurement; it is 2/2<sup>8</sup>", True),
 ("Round-3 replicates at step 1200",   "n = 5",               "n = 13 trained, 5 scored",   "Eight checkpoints verified by full restore, 2026-09-05", False),
 ("Round-3 paired &Delta;R vs multi3", "not reported",        "&minus;0.0679, t &minus;4.60, 5/5", "Training seed as unit, ticker as pairing basis", True),
 ("&hellip; its sign-flip p",          "not reported",        "0.0625 = the floor at n = 5", "2/2<sup>5</sup> lies <em>above</em> &alpha; = 0.05", True),
 ("K in every scored record",          "assumed variable",    "2, uniformly (239/239)",     "No K ladder exists, so K-dependence is unmeasurable", False),
 ("Cross-run coupling",                "assumed absent",      "present",                    "Same-seed and different-seed &Delta; intervals are disjoint", False),
 ("&hellip; my argument for it",       "&ldquo;impossible under independence&rdquo;", "withdrawn", "It happens in about half of independent designs", True),
 ("&hellip; and its effect on scoring","&ldquo;invalidates fair CRPS&rdquo;", "withdrawn",  "Within a cell the K=2 members stay exchangeable", True),
 ("Fair-CRPS bias flat in K",          "reported as measured","algebra only",               "239/239 at K=2; the magnitude&rsquo;s CI includes zero", True),
]

SECTIONS = [
 ("01", "The effect is real. The error bar is measuring the wrong thing.",
  """<p>Every registered null in the study is a <em>generation</em> null: two rollout regenerations from
  the same checkpoint. The claim is a <em>training</em> claim. Measured directly on 18 replicate
  fine-tunings that differ only in <code>--train-seed</code>, the run-to-run standard deviation of the
  panel statistic is <strong>4.46%</strong>, with a 95% interval of [2.57, 5.57].</p>
  <p>The effect it is being used to establish is <strong>+2.92%</strong>.</p>""",
  "f1_noise_floor", "Effect against its two candidate noise floors, and the per-run CRPS distribution",
  """The spread produced by re-running the fine-tuning once is larger than the difference attributed to
  the fine-tuning round. The point estimate is not the problem &mdash; twelve cross-replicate pairings of
  the same contrast give +0.0806 &plusmn; 0.0087 and never change sign. What fails is that the published
  interval prices only generation noise, the smaller of the two sources and not the one the claim varies
  over."""),

 ("02", "The p-value is the floor of its own test",
  """<p>A two-sided sign-flip permutation test on <var>n</var> paired units cannot return a value below
  2/2<sup><var>n</var></sup>. At <var>n</var> = 8 that is 0.0078125, which is exactly what all three
  contrasts report.</p>""",
  "f2_signflip_floor", "Attainable sign-flip floor against n, and the Bonferroni threshold against family size",
  """Two facts collide. The test bottoms out at 0.0078, and the study&rsquo;s own <code>family.json</code>
  enumerates 42 comparisons. Any multiplicity correction over that family puts the threshold below what the
  test can ever return, so nothing in the family survives correction &mdash; not because the effects are
  small, but because the design cannot generate enough evidence."""),

 ("03", "A confound two adversarial agents confirmed, and that does not exist",
  """<p>Two agents reported a cosine learning-rate schedule creating an 8.9&times; difference between the
  run groups at step 1200; one of them &ldquo;confirmed&rdquo; the shape
  0.5(1 + cos(&pi;s/S)). The trainer has no schedule.
  <code>wmle_full_ft.py:186</code> builds <code>optax.adamw(args.lr)</code> from a scalar, and
  <code>args.lr</code> occurs exactly once in the file.</p>""",
  "f3_no_schedule", "The assumed schedule against the constant LR in the code, and measured parameter travel",
  """Parameter travel per step over steps 1350&ndash;1500 equals travel over 150&ndash;300 to within 1.1%,
  and every run log prints <code>[Optimizer] Using inject_hyperparams (legacy scalar LR)</code>. The
  <code>cosine_anneal=True</code> they read is checkpoint metadata consumed by nothing. Adversarial review
  did not catch this. Reading the assignment did."""),

 ("04", "The peak at step 1200",
  """<p>The peak was defended with a selection correction computed as the maximum of ten independent
  normals. The statistic is (max &minus; mean)/s with s estimated from the same ten points, which is
  Grubbs&rsquo;s test, and its exact distribution is not the maximum of normals.</p>""",
  "f4_peak", "Checkpoint-position traces for both arms, and published against exact selection-corrected P",
  """Correcting the statistic makes the peak <em>more</em> significant, not less &mdash; two-fold on the
  main arm and twenty-six-fold on the control arm, which is the opposite of reassurance. The left panel
  gives the reason to distrust all of it: moving the checkpoint by one save interval changes R by up to
  0.165, which is 1.8&times; the entire round-to-round effect of 0.0904."""),

 ("05", "The variance ladder mixes two scales",
  """<p>Rungs 1 and 3 are standard deviations of <em>levels</em>. Rung 2 is the standard deviation of a
  <em>contrast</em> between two levels, which carries an extra factor of &radic;2. Comparing them directly
  overstates how far the effect sits above the noise.</p>""",
  "f5_ladder_scales", "Variance-ladder rungs on the level and contrast scales",
  """Two corrections push the same way. On a common scale the top rung is 4.08&times; the null rather than
  2.89&times;, so checkpoint position is a larger source of variation than reported. The null band itself
  is also too narrow: it was computed for a four-versus-four contrast while the headline compares two
  seeds against four, which needs a further &radic;1.5."""),

 ("06", "Five values circulate for one number",
  """<p>R for the round-4 arm at step 1200 is the quantity the headline is built on. Five different values
  for it sit in the study&rsquo;s own stores.</p>""",
  "f6_five_values", "The five circulating values of R for the round-4 arm at step 1200",
  """The four-seed mean was on disk hours before <code>sweep_curve.json</code> was written with the
  two-seed value, and the two-seed value is the one the headline used. Nothing records which estimator
  produced a stored cell: <code>crps_panel.jsonl</code> carries no provenance block, and K is stamped into
  surviving records from a shell directory count rather than from the estimator."""),

 ("07", "What more compute can and cannot buy",
  """<p>The between-round comparison has one replicate count on each side, and they are not symmetric. The
  standard error and the Welch degrees of freedom are pinned by the smaller one:
  <strong>n<sub>3</sub> = 5</strong> scored against <strong>n<sub>4</sub> = 30</strong> at step 1200.
  Eight further round-3 replicates finished on 2026-09-05; their step-1200 checkpoints restore to
  386 arrays and 159,374,987 finite elements, structurally identical to the reference, so
  n<sub>3</sub> = 13 is available as soon as they are scored.</p>""",
  "f7_sample_sizes", "Between-round standard error against each replicate count, the replicate inventory, and the paired contrast the five scored replicates support",
  """Adding round-4 replicates is nearly free of information: driving n<sub>4</sub> to infinity tightens the
  interval by 5.9% and leaves the degrees of freedom where they were. Round 3 is where the missing
  information is &mdash; n<sub>3</sub> from 5 to 13 tightens it by 23.9%.
  The right panel is why the count matters. All five scored replicates move R the same way against
  multi3, paired on ticker, with a paired t of &minus;4.60. The permutation test returns 0.0625 anyway,
  because that is 2/2<sup>5</sup> and it is the smallest value the test can produce at this n. Unanimity
  and a large t cannot clear a floor set by the sample size."""),
]

rows = "\n".join(
    f'<tr{" class=\"key\"" if key else ""}><th scope="row">{q}</th>'
    f'<td class="was">{a}</td><td class="now">{b}</td><td class="why">{w}</td></tr>'
    for q, a, b, w, key in LEDGER)

secs = []
for num, title, lede, fig, cap, reading in SECTIONS:
    secs.append(f"""
<section>
  <h2><span class="num">{num}</span>{title}</h2>
  {lede}
  <figure>
    {img(fig, cap)}
    <figcaption>{cap}</figcaption>
  </figure>
  <div class="reading"><span class="rlabel">Reading</span><p>{reading}</p></div>
</section>""")

HTML = f"""<title>The Wrong Replicate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>
:root {{
  --ground:#f6f7f8; --surface:#ffffff; --sunk:#eceef1;
  --ink:#22303f; --ink-soft:#4a5b6d; --ink-faint:#7d8b99;
  --rule:#d9dee4; --rule-soft:#e7ebef;
  --accent:#b83f2b; --accent-soft:#f0e0dc;
  --cool:#4a6f8f; --warn:#a8761a;
  --shadow:0 1px 2px rgba(34,48,63,.06), 0 8px 24px rgba(34,48,63,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#12171c; --surface:#1a2027; --sunk:#222a32;
    --ink:#dfe5ea; --ink-soft:#a9b6c2; --ink-faint:#7d8b99;
    --rule:#2c353e; --rule-soft:#242c34;
    --accent:#e87a63; --accent-soft:#3a2620;
    --cool:#8fb3d0; --warn:#d9a94e;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#12171c; --surface:#1a2027; --sunk:#222a32;
  --ink:#dfe5ea; --ink-soft:#a9b6c2; --ink-faint:#7d8b99;
  --rule:#2c353e; --rule-soft:#242c34;
  --accent:#e87a63; --accent-soft:#3a2620;
  --cool:#8fb3d0; --warn:#d9a94e;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:78rem; margin:0 auto; padding:0 1.5rem 6rem; }}
.col {{ max-width:38rem; }}
h1,h2,h3 {{ font-family:"IBM Plex Serif",Georgia,serif; text-wrap:balance; }}

header.masthead {{ padding:4.5rem 0 2.5rem; border-bottom:1px solid var(--rule); margin-bottom:3rem; }}
.eyebrow {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.7rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-faint); margin:0 0 1.1rem;
}}
h1 {{ font-size:clamp(2.1rem,5.2vw,3.4rem); font-weight:600; line-height:1.08; margin:0 0 1.2rem; letter-spacing:-.015em; }}
.standfirst {{ font-size:1.14rem; line-height:1.58; color:var(--ink-soft); max-width:40rem; margin:0 0 2rem; }}
.standfirst strong {{ color:var(--ink); font-weight:600; }}

.provenance {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:.1rem 2rem;
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.74rem;
  border-top:1px solid var(--rule-soft); padding-top:1.1rem; color:var(--ink-faint);
}}
.provenance div {{ padding:.28rem 0; }}
.provenance b {{ display:block; font-weight:500; color:var(--ink-faint); text-transform:uppercase; letter-spacing:.09em; font-size:.66rem; }}
.provenance span {{ color:var(--ink-soft); overflow-wrap:anywhere; }}

.ledger-wrap {{ margin:0 0 4.5rem; }}
.ledger-wrap h2 {{ font-size:1.5rem; margin:0 0 .35rem; font-weight:600; }}
.ledger-note {{ color:var(--ink-soft); font-size:.93rem; margin:0 0 1.4rem; max-width:38rem; }}
.tscroll {{ overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:3px; box-shadow:var(--shadow); }}
table {{ border-collapse:collapse; width:100%; min-width:44rem; font-size:.86rem; }}
thead th {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:500; font-size:.68rem;
  letter-spacing:.11em; text-transform:uppercase; color:var(--ink-faint);
  text-align:left; padding:.9rem 1rem; border-bottom:1px solid var(--rule); white-space:nowrap;
}}
tbody th {{ text-align:left; font-weight:500; padding:.72rem 1rem; color:var(--ink); }}
tbody td {{ padding:.72rem 1rem; vertical-align:top; }}
tbody tr + tr th, tbody tr + tr td {{ border-top:1px solid var(--rule-soft); }}
td.was, td.now {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.was {{ color:var(--ink-faint); text-decoration:line-through; text-decoration-color:var(--ink-faint); text-decoration-thickness:1px; }}
td.now {{ color:var(--accent); font-weight:500; }}
td.why {{ color:var(--ink-soft); font-size:.82rem; }}
tr.key th {{ box-shadow:inset 3px 0 0 var(--accent); }}

section {{ margin:0 0 4.75rem; }}
section h2 {{ font-size:1.72rem; font-weight:600; line-height:1.22; margin:0 0 1rem; max-width:34rem; letter-spacing:-.008em; }}
.num {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.72rem; font-weight:500;
  color:var(--accent); letter-spacing:.1em; display:block; margin-bottom:.5rem;
}}
section p {{ max-width:38rem; margin:0 0 1rem; }}
code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.86em; background:var(--sunk); padding:.1em .36em; border-radius:2px; }}
var {{ font-family:"IBM Plex Serif",Georgia,serif; font-style:italic; }}

figure {{ margin:2rem 0 1.4rem; }}
figure img {{ width:100%; max-width:100%; height:auto; display:block; background:#fff; border:1px solid var(--rule); border-radius:3px; }}
figcaption {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.71rem; color:var(--ink-faint);
  margin-top:.7rem; max-width:40rem; line-height:1.5;
}}
.reading {{ border-left:2px solid var(--accent); padding:.15rem 0 .15rem 1.15rem; max-width:38rem; }}
.rlabel {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.66rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--accent); display:block; margin-bottom:.3rem;
}}
.reading p {{ margin:0; color:var(--ink-soft); }}

.close {{ border-top:1px solid var(--rule); padding-top:2.6rem; }}
.close h2 {{ font-size:1.5rem; margin:0 0 1.4rem; }}
.close h3 {{ font-size:1rem; font-weight:600; margin:1.8rem 0 .45rem; color:var(--ink); }}
.close p {{ max-width:38rem; color:var(--ink-soft); margin:0; }}
footer {{ margin-top:3.5rem; padding-top:1.4rem; border-top:1px solid var(--rule-soft);
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.71rem; color:var(--ink-faint); }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>

<div class="wrap">
<header class="masthead">
  <p class="eyebrow">Statistical audit &middot; sigma-0 &middot; 2026-09-05</p>
  <h1>The wrong replicate</h1>
  <p class="standfirst">A fine-tuning study concluded that round&nbsp;4 was worse than round&nbsp;3, carried by a
  sign-flip test at <em>p</em>&nbsp;=&nbsp;0.0078. Six adversarial agents and seven solution agents attacked it.
  <strong>The point estimate survived. The inference did not</strong> &mdash; every error bar in the study was
  measured on generation replicates, while the claim varies the training run, and that spread is larger than
  the effect. Then three of this audit&rsquo;s own claims failed a second check, and are withdrawn below.</p>
  <div class="provenance">
    <div><b>Repository</b><span>KangOxford/sigma-0 (private)</span></div>
    <div><b>Branch</b><span>audit/crps-return-alignment-20260905</span></div>
    <div><b>Commit</b><span>0d7b0f0a468352c9c2cc7cc64411085ff6dca95c</span></div>
    <div><b>Pull request</b><span>#76</span></div>
    <div><b>Notebook</b><span>12 code cells, 0 errors, 9 figures</span></div>
    <div><b>Replicates measured</b><span>18 fine-tuning runs</span></div>
    <div><b>Round-3 replicates</b><span>13 trained &middot; scoring in progress</span></div>
  </div>
</header>

<div class="ledger-wrap">
  <h2>What the audit changed</h2>
  <p class="ledger-note">Seventeen quantities moved. The rows with a marked rule change what the study can
  claim; the rest is arithmetic that was already pointing the same way. The last three are claims this
  audit made and then had to withdraw.</p>
  <div class="tscroll">
    <table>
      <thead><tr><th scope="col">Quantity</th><th scope="col">As published</th><th scope="col">Corrected</th><th scope="col">Why it changed</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>

{"".join(secs)}

<div class="close">
  <h2>What this leaves standing</h2>
  <h3>The measurement is fine; the inference was scoped to the wrong replicate.</h3>
  <p>The point estimate of the round-4 minus round-3 contrast is stable across every re-pairing of the data.
  What was never estimated is the variability of training a model twice, and that turns out to be the larger
  term. On the corrected interval the panel effect is not distinguishable from zero.</p>
  <h3>The design cannot express the claim at the resolution it was reported.</h3>
  <p>Eight tickers give a sign-flip floor of 0.0078, the study&rsquo;s own family has 42 comparisons, and no
  correction over that family can leave anything alive. Reporting <em>p</em> = 0.0078 in that setting states
  that all eight units agreed, not that the effect is large.</p>
  <h3>Checkpoint position dominates.</h3>
  <p>One save interval moves R by up to 0.165, against a round-to-round effect of 0.090. Any comparison
  between arms at a single hand-picked step is reading a quantity whose largest source of variation is which
  step was picked.</p>
  <h3>The thirteen units are independent replicates of a conditional experiment.</h3>
  <p>The first version of this answered by comparing launch flags &mdash; same parent path, same weights
  file, same item-set hash &mdash; which proves nothing, so it was checked by measurement instead. The
  parent has not been touched since 2026-08-14 and all thirteen checkpoints carry the <em>parent&rsquo;s
  own</em> <code>init_timestamp_nsecs</code>, so the weights are restored and never initialised;
  recomputing <code>default_rng(train_seed).permutation(4800)</code> reproduces the order hash and first
  eight indices each run logged, for three seeds from each family; and any two runs share 31.18% of their
  training items against an independent-subsampling expectation of 31.25%.</p>
  <p>That last figure is <strong>evidence for independent streams, not against them</strong>, and an
  earlier revision of this page had it the other way round. A shared parent does not make the units
  dependent on one another &mdash; it defines what they are conditional <em>on</em>. The estimand is
  Var(R | &theta;<sub>0</sub>, D, C, G, B): the restored parent, the 4800-item pool, the 500 contexts,
  the generation seeds 97901/97902, and the 1500-step budget.</p>
  <p>Also withdrawn: calling this a <em>lower bound</em> on training-run variance. The law of total
  variance makes the <em>average</em> conditional variance at most the marginal, but a single
  conditioning set can sit above or below that average, so one fixed parent bounds the marginal in
  neither direction. The genealogy audit narrows the scope of the conclusion and confirms the
  randomisation is sound; it is not a refutation of the design.</p>
  <h3>Three of my own claims did not survive their own check.</h3>
  <p>I wrote that a coupled <em>D&#772;</em> below <em>W&#772;</em> was impossible for independent draws. It
  is not: when runs share a distribution the two expectations are equal, so the difference is a coin
  flip, and a simulation of independent designs of this shape lands below zero about half the time. The
  coupling is real, but the evidence is the gap between same-seed and different-seed pairings, not the
  sign. I also wrote that shared generation seeds invalidate the scoring; they do not touch the
  within-cell estimator, whose two members are exchangeable given the checkpoint, and the variance
  reduction they are supposed to buy on paired comparisons measures 0.4% here. And &ldquo;the bias is
  flat in K&rdquo; is algebra about which estimand you are aiming at, reported as though it had been
  measured &mdash; every one of the 239 records is at K = 2, so no K-dependence is estimable at all.</p>
  <h3>The verdict at the sample size in hand: underpowered, and confounded.</h3>
  <p>Neither supported nor refuted, for three reasons each sufficient alone. At n<sub>3</sub> = 5 the
  permutation test bottoms out above &alpha;. The contrast reads replicates at step 1200 against multi3 at
  its endpoint, so it carries the checkpoint-position term that moves R by 2.4 times the contrast itself.
  And round 1 and the parent cannot enter the comparison at all: parent_multi2 is scored on one ticker and
  wm_ft_multi on none &mdash; a gap in what was scored, not in compute.</p>
  <h3>One methodological note that outlives this study.</h3>
  <p>An adversarial reviewer confirmed a learning-rate schedule that does not exist in the code, from a
  metadata field, and a second agent reproduced its arithmetic. Reading the value that was recorded rather
  than the value that was used is a failure mode this codebase has hit before, and adversarial review did not
  protect against it &mdash; only reading the assignment did.</p>
  <footer>Every figure is generated by the notebook from files on disk. Sources:
  variance_ladder.json, measured_spreads.json, crps_panel.jsonl, fix_attribution.json,
  and wmle_full_ft.py.</footer>
</div>
</div>
"""

out = O / "crps_audit_artifact.html"
out.write_text(HTML)
print(f"wrote {out}: {out.stat().st_size/1024:.0f} KB")
