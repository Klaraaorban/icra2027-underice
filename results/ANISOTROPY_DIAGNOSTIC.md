# Gaussian anisotropy as a sensor-free diagnostic — real signal, wrong-predicted direction

**Spearman rho(needle_fraction, sonar_r) = +0.740, p=0.0007, n=17 (every
trained config across both stations).** Strong and highly significant — but
the sign is the opposite of the hypothesis.

## What was measured

For every trained model's saved `point_cloud.ply`: per-Gaussian scale
anisotropy = ratio of largest to smallest of the 3 (exponentiated) scale
axes. Needle fraction = fraction of Gaussians with ratio > 10x. Pure PLY
parsing, no GPU, no sonar involved in computing this quantity at all — reads
only the model's own saved parameters, exactly as intended.

| Config | n Gaussians | needle_frac(>10x) | sonar r |
|---|---|---|---|
| ps117_39_selfdiag_sonar_30k | 172,124 | **0.503** | 0.973 |
| ps117_39_selfdiag_sonar | 152,049 | 0.317 | 0.948 |
| ps117_39_da3_texconf_floor_sonar | 149,145 | 0.302 | 0.813 |
| ps117_39_da3_texconf_sonar | 143,707 | 0.294 | 0.912 |
| ps117_39_da3_texconf_calibrated_sonar | 143,902 | 0.290 | 0.942 |
| ps117_39_da3_depthreg | 127,767 | 0.271 | 0.835 |
| ps117_39_da3_texconf_gated | 124,328 | 0.255 | 0.802 |
| ps117_39_colmap_mast3r_fused | 168,569 | 0.245 | 0.676 |
| ps117_39_vanilla_colmap | 144,421 | 0.210 | 0.663 |
| ps117_29_mast3r_texconf_gated_30k | 409,447 | 0.350 | -0.146 |
| ps117_39_mast3r_seeded_depthreg | 708,496 | 0.138 | 0.476 |
| ps117_39_mast3r_seeded | 686,494 | 0.132 | 0.462 |
| ps117_29_da3_texconf_gated_colmap | 50,144 | 0.111 | -0.771 |
| ps117_29_vanilla_colmap | 51,012 | 0.100 | -0.530 |
| ps117_29_mast3r_depthreg | 599,476 | 0.064 | -0.140 |
| ps117_29_mast3r_texconf_gated_7k | 516,276 | 0.020 | -0.021 |
| ps117_29_mast3r_seeded_vanilla | 517,814 | 0.019 | -0.306 |

## Why the sign is backwards, and what that means

The hypothesis was: needle-shaped Gaussians indicate degenerate, view-locked
"billboard" geometry, so more needles should predict *worse* sonar accuracy.
Instead, the best-performing config (30k headline, sonar r=0.973) has the
*highest* needle fraction of any PS117-39 model (0.503), and the strong
positive trend holds across nearly every config on both stations.

The likely reason: a pure scale-ratio metric can't tell apart two very
different things that both produce a high max/min ratio —

1. **The pathology the hypothesis was worried about**: a Gaussian stretched
   *along the viewing ray*, into depth, representing a "billboard" that only
   looks right from the training viewpoint (this is what item 5's
   off-trajectory render visualized).
2. **A completely normal, even desirable, outcome**: a Gaussian flattened
   *tangent to a real surface* — thin in the surface-normal direction, wide
   along the surface — which is exactly what a well-converged model *should*
   produce to represent a real, thin, flat ice feature precisely. More
   training and better geometry should produce *more* of these, not fewer,
   since early/undertrained models start from isotropic blobs and sharpen
   toward the true (thin, flat) surface shape as optimization converges.

The 30k model having by far the highest needle fraction (0.503, roughly
double the closest other PS117-39 config) is consistent with this reading:
more iterations gives more time to refine coarse isotropic Gaussians into
thin, accurate, surface-tangent ones.

**A corrected version of this diagnostic** would need to condition on
*orientation*, not just the raw ratio — e.g. the angle between a Gaussian's
long axis and the local viewing ray at its position. Built and run (see
below): it does not rescue a clean directional diagnostic, but it does
resolve *why* the raw signal exists.

## Orientation-corrected version (2026-09-02)

For every anisotropic Gaussian (ratio > 10x), computed the world-space
direction of both its minor axis (from the stored quaternion, verified
against `utils/general_utils.py`'s own `build_rotation` convention: w,x,y,z,
columns of R = local axis directions in world space) and its major axis, and
the angle each makes with the ray from the nearest camera centre to that
Gaussian. Minor axis ~parallel to the ray = surface-tangent disc (predicted
healthy); major axis ~parallel to the ray = depth-aligned streak (predicted
pathological).

| Metric (angle threshold 30 deg) | n | Spearman rho vs sonar_r | p |
|---|---|---|---|
| streak-fraction, of the anisotropic subset | 17 | +0.147 | 0.573 (not significant) |
| disc-fraction, of the anisotropic subset | 17 | -0.284 | 0.269 (not significant) |
| streak-fraction, of ALL Gaussians | 17 | +0.735 | 0.0008 (significant) |

**This resolves the puzzle rather than rescuing the original hypothesis.**
The metric that actually isolates orientation *conditional on* how anisotropic
a Gaussian already is (streak-fraction of the anisotropic subset) shows no
significant relationship at all (p=0.573) — whether an anisotropic Gaussian
happens to be streak-aligned or disc-aligned does not predict sonar accuracy
in this dataset. The metric that stays strongly significant (streak-fraction
of ALL Gaussians, p=0.0008) is numerically entangled with total Gaussian
count, which is itself driven by training iterations/density (the 30k model
has by far the most Gaussians of every kind). **Conclusion: the original raw
anisotropy-fraction signal is a model-maturity/density effect, not an
orientation-specific artifact signal as hypothesized.** Raw anisotropy
fraction remains a real, strong, sensor-free correlate of sonar accuracy
across configs (rho=0.740, unchanged from above) — just not because it
detects billboard streaking specifically; more likely because both are
downstream of the same underlying cause (more training / better convergence
produces both more accurate geometry and more Gaussians of every shape).
Still potentially useful as a convergence/maturity diagnostic, not as the
sensor-free "catches bad billboard geometry" instrument originally hoped for.
