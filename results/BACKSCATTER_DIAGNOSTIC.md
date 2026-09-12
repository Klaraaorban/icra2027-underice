# Backscatter-as-geometry — real signal, but not cleanly isolated

**Inconclusive as measured, not negative.** A real, measured elevation exists
but the metric as built doesn't cleanly separate "water-column backscatter"
from ordinary low-opacity Gaussians every 3DGS model accumulates.

## What was measured (PS117-39 30k headline model, 172,124 Gaussians)

- 80.7% of ALL Gaussians in the model are low-opacity (<0.3) — a surprisingly
  high baseline, before even considering proximity to the camera path.
- Gaussians within 0.3 COLMAP units (~1.57m, using the fitted 5.24 scale) of
  any camera centre are low-opacity 99.6% of the time, vs. 80.5% far from any
  camera — a real, consistent elevation, but only a **1.24x** ratio on top of
  an already very high baseline.

## Why this is inconclusive rather than a clean confirmation

A 1.57m "near-camera" radius, given the whole reconstructed ice-underside
region only spans a few metres, ends up covering a large fraction of the
scene's useful volume, not a tight halo specifically in front of the lens.
This dilutes whatever water-column-specific signal exists with the ordinary
population of low-opacity soft/anti-aliasing Gaussians that accumulate at
depth discontinuities and thin surface edges in any 3DGS reconstruction,
water or no water. The 80.7% overall baseline is itself surprisingly high and
worth separate scrutiny (it may indicate general over-densification in this
recipe, independent of the water-backscatter question).

**What would actually test the hypothesis**: distance along each camera's
individual viewing RAY, restricted to a tight radius (e.g. <20cm) and only
within the camera's frustum near-plane, rather than Euclidean distance to any
of 300 overlapping camera centres pooled together. That's a per-camera,
per-ray computation, not the simple pooled-distance one built here — flagged
as the concrete next step rather than attempted in this pass, since the
current implementation's ambiguity means an ablation (cull these Gaussians,
re-render, compare PSNR/sonar r) would be testing the wrong Gaussians and
risks a misleading answer either way.
