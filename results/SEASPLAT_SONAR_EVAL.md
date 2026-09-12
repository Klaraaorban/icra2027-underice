# SeaSplat vs. real sonar — how bad is the un-sonar-supervised baseline?

**Answers: "if I added sonar to SeaSplat, how much headroom is there?"**

## Setup

Used the existing trained SeaSplat checkpoint at
`gaussian-splatting/data/ps117_39_30to90_fisheye/experiments/08272026/test`
(iteration 7000, no retraining) — PS117-39, same source dataset as every
other number in this project. SeaSplat has **zero sonar supervision**; this
is its raw underwater-physics-model (attenuation/backscatter-aware) geometry,
straight out of the box.

Same rigor as every other config: scale factor fit on TRAIN split only
(least-squares, no intercept), scored on the held-out TEST split, both
Pearson r and RMSE reported. Script: `scripts/seasplat_sonar_eval.py`.

**Caveat, disclosed rather than buried**: SeaSplat's checkpoint used its own
default `eval=True` train/test split (interleaved, not this project's later
contiguous-block holdout). This is the same weaker-holdout convention this
project moved away from for its own headline numbers — reused here only
because re-splitting would mean retraining, which the user did not ask for.
Treat this as a first-look number, not a table-ready apples-to-apples entry
until re-run with a matched holdout if it goes in the paper.

## Result

| | n | Pearson r | RMSE |
|---|---|---|---|
| Train | 260 | **-0.865** | 1.354 m |
| Test (held-out) | 38 | **-0.861** | 1.340 m |

Fitted scale: 0.188 (i.e. SeaSplat's raw depth units are ~5.3x the real
metric scale before correction — expected, since nothing in its pipeline
ever saw the sonar to calibrate scale).

Our best sonar-supervised result on this same station: **RMSE = 0.181 m**.

**SeaSplat's raw geometry is ~7.4x worse in RMSE, and its correlation with
real sonar depth is not just weak — it's strongly *negative*.** Train and
test agree closely (-0.865 vs -0.861), so this isn't overfitting noise; it's
a stable, reproducible property of this checkpoint: as real range increases,
SeaSplat's estimated depth systematically decreases.

## What this means for "should I add sonar to SeaSplat"

This is a stronger case for adding sonar supervision than our own pipeline
ever was. Our headline pipeline, even *before* any sonar supervision, mostly
showed weak-to-moderate correlation with real depth (positive in favorable
regions, only the known PS117-29 anomaly region showed this kind of sign
inversion). SeaSplat here shows the PS117-29-style catastrophic sign
inversion as its *default, everywhere* behavior on PS117-39 — a station
where our own DA3-seeded pipeline does not have this problem.

That points to the cause being architectural, not scene-specific: SeaSplat's
SfM initialization + backscatter/attenuation optimization has no mechanism
to anchor absolute or even relatively-signed depth without an external
metric reference — exactly the gap sonar supervision is designed to fill.
Given how large and consistent the sign inversion is, adding sonar (even a
modest weight, per the sensitivity sweep pattern already established) should
have a *large* effect here — likely bigger than what it bought our own
pipeline, precisely because there's so much more room to fix.

## Not done in this pass

- Re-running with a contiguous-block holdout matched to the rest of the
  project's numbers (would need retraining or at least a rebuilt eval split).
- Diagnosing *why* the sign flips (SeaSplat's `render_depth` convention,
  its SfM init, or the backscatter/attenuation model itself) — same class of
  investigation as the PS117-29 wrong-sign root-causing, not run here since
  the user's question was the number, not the mechanism.
- Actually adding sonar supervision to SeaSplat's training loop (would need
  porting the sonar loss term into SeaSplat's `train.py`, a real
  implementation task, not an eval one).
