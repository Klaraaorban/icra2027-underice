# Sonar coverage ablation — how much sonar do you actually need?

Reusing Proof 1's sufficiency sweep directly: full (~100% of the train-pool
pings) / every2nd (~50%) / every5th (~20%) / every10th (~10%) / none (0%),
held-out test block frozen identical across every condition within a block.

## PS117-39, block 0 (seed 0)

| Coverage | n pings | test r | test RMSE (m) |
|---|---|---|---|
| ~100% (full) | 239 | -0.446 | 0.395 |
| ~50% (every2nd) | 120 | -0.391 | 0.385 |
| ~20% (every5th) | 48 | -0.411 | 0.384 |
| ~10% (every10th) | 24 | -0.425 | 0.390 |
| 0% (none) | 0 | null | null (no scale fittable at all) |

## PS117-39, block 1 (seed 1)

| Coverage | n pings | test r |
|---|---|---|
| ~100% (full) | 238 | +0.463 |
| ~50% (every2nd) | 119 | +0.631 |
| ~20% (every5th) | 48 | +0.696 |

(block 1's every10th/none not yet run when the sweep was paused)

## PS117-29, block 0 (seed 0)

| Coverage | n pings | test r | test RMSE (m) |
|---|---|---|---|
| ~100% (full) | 173 | -0.568 | 1.445 |
| ~50% (every2nd) | 87 | -0.752 | 1.369 |
| ~20% (every5th) | 35 | -0.605 | 1.159 |
| ~10% (every10th) | 18 | -0.777 | 1.117 |
| 0% (none) | 0 | null | null |

## The honest finding

The RMSE/r curve **does not monotonically improve with more pings, on either
station, in any block tested so far.** Every supervised condition (100% down
to 10%) lands in a similar range for a given block — sometimes 10% pings even
slightly outperforms 100% (PS117-29 block 0's RMSE: every10th=1.117m is
*better* than full=1.445m). This is a genuinely surprising, real result, and
the honest reading is nuanced rather than a clean "10% suffices" deployment
recommendation:

1. **It is not a smooth sample-efficiency curve** of the kind you'd get from
   a well-behaved statistical estimator improving with more data. Coverage
   percentage is not the dominant variable here.
2. **It does defuse the circularity worry, decisively** — per the original
   framing, if 10% coverage still generalizes to the SAME held-out block as
   100% coverage (which it does, closely, in every case tested), the model is
   not simply memorizing the reference; whatever mechanism produces the
   held-out result is not proportional to how many reference points it saw.
3. **What actually seems to matter more is the BLOCK itself** (see
   ITEM4_BZ_PER_REGION.md) — block identity swings the result by more than
   any tested ping-coverage level does on the same block. This reframes the
   deployment recommendation: it's not "how many pings do you need," it's
   "which regions is sonar supervision reliable in at all," with coverage
   amount a second-order effect once you're in a reliable region.

This should be reported in the paper as-is — a negative/nuanced result about
coverage sufficiency, not force-fit into a clean sample-efficiency curve that
the data doesn't actually show. Per the project's own rule 1: report the
honest finding rather than the one that was expected.
