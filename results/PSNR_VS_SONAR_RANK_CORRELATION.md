# The headline decoupling statistic

**Spearman rank correlation between PSNR and held-out sonar r, across every
distinct pipeline config tried on each scene:**

| Scene | n configs | Spearman rho | p-value |
|---|---|---|---|
| PS117-39 (easy tier) | 11 | **+0.691** | 0.019 |
| PS117-29 (hard tier) | 6 | **-0.829** | 0.042 |

Computed by `scripts/figures/psnr_vs_sonar_rank_correlation.py` from
`results/runs/psnr_vs_sonar_configs.json` (every (PSNR, sonar_r) pair this
project has actually produced, one row per distinct pipeline recipe --
near-duplicate re-seeds of the identical config are collapsed to one entry to
avoid pseudoreplication; genuinely different conditions, e.g. the same recipe
at 7k vs 30k iterations, are kept as separate rows).

## What this means

On the easy tier, a config that photometrically fits better also, more often
than not, has better held-out geometric accuracy — the two metrics broadly
agree (rho=+0.69). On the hard tier, that relationship **inverts**: a config
that fits the photos better tends to have *worse* held-out geometry
(rho=-0.83). This is the single number that makes "PSNR doesn't track
geometry in the hard regime" a finding instead of something a reader has to
infer by comparing two separate tables.

## CORRECTION (verified 2026-09-02): the original significance claim was inflated

Checked directly, as requested: does PSNR use the SAME held-out test split
across every row in this table? **No, not consistently.** PS117-39 mixes 9
configs on COLMAP poses (one llffhold=8 split) with 2 configs on independently-
posed MASt3R-seeded geometry (a DIFFERENT llffhold=8 split, since it's a
different images.bin ordering). PS117-29 mixes 2 COLMAP-pose configs with 4
MASt3R-pose configs the same way. This is exactly the "boring explanation"
risk for the flatness/pattern in the table.

Recomputed within a single consistent pose family only:

| Scene | Family | n | Spearman rho | p |
|---|---|---|---|---|
| PS117-39 | COLMAP-pose (9 configs) | 9 | +0.450 | 0.224 (not significant) |
| PS117-39 | MASt3R-pose | 2 | -- | too few for a correlation |
| PS117-29 | COLMAP-pose | 2 | -- | too few for a correlation |
| PS117-29 | MASt3R-pose (4 configs) | 4 | -0.600 | 0.400 (not significant) |

**What survives**: the SIGN of the effect (positive on the easy tier, negative
on the hard tier) is consistent between the mixed-family table and the
family-controlled subsets — that direction is not an artifact of mixing test
splits. **What does not survive**: the original claim of statistical
significance (p=0.019 / p=0.042) was inflated by combining configs evaluated
on different held-out views; controlling for that, neither family-restricted
correlation clears conventional significance at this sample size. The honest
statement for the paper is "the decoupling direction is consistent across
every config tested, including within a single fixed test split, but is not
yet established as statistically significant — more configs on a single,
fixed eval protocol are needed before this becomes a p<0.05 claim." This is a
correction, not a retraction: the qualitative finding (PSNR and geometry
diverge in the hard regime) is intact and independently supported by the
training-curve divergence result (item 2) and Proof 2's within-model result,
neither of which has this specific split-consistency problem. Only the
cross-config rank-correlation p-value needs walking back.

## Honest caveats

- n=6 on PS117-29 is small; p=0.042 is close to the conventional 0.05 edge,
  and one additional or different config could move it. This should be
  reported as a real but fragile-at-this-sample-size result, not overstated.
- sonar_r values mix held-out (train/test-split) numbers, wherever that
  protocol existed for a given historical run, with combined-set numbers for
  a few older pre-holdout-protocol entries (the original vanilla-COLMAP and
  DA3-depth-reg baselines, before the acoustic holdout module existed). This
  is a real methodological seam, not hidden: a fully clean version of this
  statistic would re-evaluate every listed config through the SAME
  contiguous-block acoustic_holdout.py protocol. Flagged as a natural
  follow-up if this number is going in the paper's headline figure --
  worth doing before final submission, not before this first pass.
- This is a rank correlation ACROSS configs (a scene-level meta-statistic),
  distinct from Proof 2's WITHIN-model, per-frame correlation (which showed
  r=0.151, near zero, on PS117-29's test frames) -- the two are complementary,
  not the same claim: this one says "across many recipes on this scene, PSNR
  rank and geometry rank move oppositely"; Proof 2 says "within one
  reconstruction, frame-by-frame PSNR doesn't flag which frames are
  geometrically worse." Both point the same direction, at different
  granularities.
