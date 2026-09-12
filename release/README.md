# What this is

We're building a 3D reconstruction method for under-ice ROV footage
(Antarctic, real cruise data) for an ICRA 2027 submission. The core idea:
vision-only 3D reconstruction (3D Gaussian Splatting) doesn't know true
scale or depth on its own, especially on low-texture ice where many
different 3D shapes explain the same photos equally well. We add a cheap
single-beam sonar (altimeter) as a training signal, and the reconstruction
self-checks its own agreement with it.

**This folder answers one question: does that actually help, and by how
much?** It's a deliberately narrow slice of a much larger project — just
enough to explain and reproduce that one result, not the whole research
process.

- Start with **`Sonar_Story.pdf`** (2 pages) — the goal, the result, and a
  direct visual sanity check, in plain language.
- **`sonar_eval/`** is the code that produced the numbers in it: three short
  scripts, one per method compared, plus a README on how to run them.

## Headline result

| Method | Sonar r (mean ± std, 5-fold CV) | Sonar RMSE |
|---|---|---|
| Ours (sonar-supervised) | +0.980 ± 0.013 | 0.075 ± 0.022 m |
| SeaSplat (no sonar) | -0.234 ± 0.771 | 1.460 ± 0.397 m |
| WaterSplatting (no sonar) | -0.245 ± 0.754 | 1.416 ± 0.387 m |

Both baselines show consistent sign-inverted correlation on their training
frames (r ≈ -0.85 across all blocks), but the scale fitted on that inverted
training data is spatially inconsistent — so test-block r varies widely,
including flipping positive in some blocks. Full explanation in the PDF.
