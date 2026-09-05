# Adversarial review brief — your job is to find what is wrong, not to summarise

Five agents drafted a plan for continual learning on sigma-0. You are one of five
reviewers, each with a different lens. Read every draft in
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/`:

    _BRIEF.md              the shared facts the drafters were given
    01_measurements.md     the measurement ladder and decision rules
    02_statistics.md       estimands, resampling, noise floor, power, multiplicity
    03_infrastructure.md   preconditions, manifests, profiling, delete/refactor
    04_training_design.md  arms, rewarm, pool construction, assertions
    05_execution.md        dependency graph, sizing, attach recipe, ordering

Then read enough of the actual code and data to check the drafts against reality rather
than against each other. Paths worth reading are named in `_BRIEF.md`.

## What a finding must contain

A finding is only useful if it is falsifiable and actionable. Each one needs:

- **the claim you are attacking**, quoted, with the file and section it came from
- **why it fails**, mechanically, not "this seems risky"
- **the concrete scenario** in which the plan produces a wrong answer or cannot run
- **the cheapest check** that would settle whether you are right, ideally a CPU command
- **severity**: BLOCKING (the plan produces a wrong conclusion or cannot execute) /
  MAJOR (wastes substantial compute or leaves a claim unsupported) /
  MINOR (imprecision worth fixing)

Rank BLOCKING first. An empty finding list is an acceptable answer for a lens that
genuinely found nothing, and is far better than padding — but say what you checked.

## Contradictions between drafts are first-class findings

The five drafters could not see each other's work. Where two drafts assume different
things about the same object — the same score, the same sample size, the same file, the
same decision threshold — that is a real defect in the merged plan, and you should report
it even if each draft is individually defensible.

## Defects this project has actually shipped; check for each

- A knob set, printed and recorded that never reached the code (`TOKEN_MODE` pinned to
  26tok in five places, four silently; the only symptom was a derived quantity).
- A default value that never applied because something downstream overwrote it.
- A metric whose name is not its semantics.
- Dividing by a per-group constant: shrinks noise, does not shrink bias, manufactures
  significance. `t` grew from 0.66 to 36.61 on a pure accounting change.
- An effect that shrank as n grew, because the power calculation reused an effect size
  estimated from the same small sample.
- A whole line claimed from one measured slice, when another slice was minutes away.
- A selection rule's consequence reported as a property of the data.
- A null control that shares its error with the treatment, so it is not null.
- A noise floor estimated on a different structure from the effect (unpaired vs paired),
  or from a single group.
- A verdict read at a quantile carrying 8 events.
- A comparison whose two arms were evaluated on positions only one of them trained on.
- A self-test that cannot fail because `from x import NAME` early-binds past the patch.
- An improvement that a known mechanical relationship already implies, credited to
  training.

## Environment facts that make plans unrunnable here

Isambard-AI on Lustre. `find` from broad roots, `ls -R`, `ls -1td`, `du -sh`, `tree` and
deep globs are forbidden outright. `rm` is forbidden; rename with a timestamp. `scancel`
is forbidden except on provably dead training. Checkpoints go to node-local storage and
are rsynced back. `--gres=gpu:1` binds whatever device Slurm picks rather than a free one.
`--cpu-bind=none` is required to attach to a node whose CPUs are busy. `--export=ALL`
carries the login node's `TMPDIR` into the step. An `srun` client lives on the login node
and dies with the session. `srun` inside a `while read` loop eats the loop's stdin. The
project inode quota has been hit at 51.2M/51.2M and renaming does not free inodes. Node-local
artefacts vanish when the allocation expires, and an expired allocation refuses `srun`.

## Output

Write your findings to the path given in your task, in **English**, opening with a short
`## 中文速览`. Then a findings table (severity, draft file, claim, defect), then one
section per finding with the detail above. End with `## What I checked and found clean`,
so the merge step knows which lenses were actually exercised.
