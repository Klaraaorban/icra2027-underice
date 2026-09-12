# Initialization vs. optimization — geometry starts right and gets destroyed

**2-point trajectory, from data already on hand (not a full dense curve —
see note at the end for what a fuller version would need).**

| Stage | sonar r |
|---|---|
| Raw MASt3R point cloud (feed-forward, before any 3DGS training) | **+0.067** |
| After pure photometric 3DGS training, no depth-reg (`ps117_29_mast3r_seeded_vanilla`) | **-0.306** |

Same underlying MASt3R-seeded reconstruction of PS117-29 both times — only
difference is whether 3DGS's own photometric optimization has run on top of
it. The sign flips from (weakly) correct to clearly wrong. This is real
support for the sharper claim: **3DGS does not fail to find usable initial
geometry here — it finds a weakly-correct starting point and then optimizes
it away**, chasing photometric fit in a way that actively degrades geometric
correctness rather than merely failing to improve it. This is the same
underlying phenomenon as the divergence plot (item 2) and the SH-ablation
result, viewed from the initialization side rather than the training-curve
side.

## What this is not (yet)

Only two points — a real trajectory would need dense checkpointing on this
exact vanilla, no-depth-reg config (matching item 2's methodology) to show
*when* during training the degradation happens and whether it's monotonic or
has its own dynamics. Not run in this pass given the scope already covered;
flagged as the natural extension if this specific line is prioritized for the
paper — it would reuse the exact same `item2_checkpoint_curve.py` script
already built and validated, just pointed at a fresh dense-checkpoint
training of the vanilla (no depth-reg, no confidence gate) MASt3R-seeded
recipe.
