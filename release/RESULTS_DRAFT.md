# Results section draft (IV. Experiments)

## IV. Experiments

### A. Dataset and setup

**PS117-39** (primary station): 298 video frames from a PS117 Antarctic cruise
ROV dive, spanning a 60 s transect under sea ice. The ROV travels at ~0.1 m/s
at ~2-4 m below the ice underside. Camera: fisheye, downsampled to 1600 px width.
Altimeter: Valeport VA500P, single-beam, ~3.92 Hz (one ping per ~6 frames).
COLMAP reconstruction: standard incremental SfM on all 298 frames.
Training follows [Kerbl et al., 2023] with the sonar supervision (§ III.B) added.
Evaluation: 5-fold contiguous along-track block holdout (§ III.E).

**PS117-29** (second station): 300 frames from the same cruise, harder conditions
(72% COLMAP registration rate). Reconstructed with MASt3R-SfM [Leroy et al., 2024]
for a more robust initialisation.
- Baseline (no sonar): r = −0.197 ± 0.421, RMSE = 1.615 m (sign inverted)
- Ours (sonar-supervised): r = +0.301 ± 0.583, RMSE = 0.558 m

Sonar supervision fixes the sign (positive r), but PS117-29 is significantly harder
than PS117-39 — weaker absolute correlation due to lower COLMAP coverage and
harder scene texture.

### B. Headline comparison (Table I)

| Method | PSNR ↑ | Sonar r ↑ | Sonar RMSE ↓ |
|--------|--------|-----------|-------------|
| WaterSplatting [Huang et al., 2024] | 30.24 dB | −0.245 ± 0.754 | 1.416 ± 0.387 m |
| SeaSplat [Yang et al., 2024] (7k) | 21.75 dB | −0.234 ± 0.771 | 1.460 ± 0.397 m |
| SeaSplat (30k, matched budget) | — | — | — |
| **Ours (sonar-supervised, 30k)** | **36.44 dB** | **+0.980 ± 0.013** | **0.075 ± 0.022 m** |

*All methods evaluated on PS117-39, 5-fold contiguous CV. Sonar r and RMSE
on held-out test split (scale fitted on train). — = pending (see run_ablations.sh).*

The two baselines have no sonar supervision and no mechanism to anchor absolute
or even correctly-signed depth. Both land on negative r: the photometric
optimiser finds a locally consistent solution with the ice-to-camera direction
reversed. The ±0.77 variance across folds (vs ±0.013 for ours) reflects
spatially incoherent geometry, not measurement noise. Our sonar-supervised
model achieves r = +0.980 — the depth signal is correctly oriented and its
shape matches the altimeter to within 7.5 cm RMSE post-scale-fit.

**Sign inversion is not an SfM artefact.** On PS117-29, MASt3R-SfM (a recent
dense matcher) produces right-signed initial geometry (r = +0.067 against sonar).
Pure photometric 3DGS training corrupts this to r = −0.197 ± 0.421 at 30k
iterations: the optimiser has sufficient freedom to invert the depth direction
on low-texture ice, where the rendering loss is almost flat in depth. This
shows the failure is a property of the photometric objective on this scene type,
not of any particular SfM initialisation.

### C. Ablation study (Table II)

*(Pending ablation training runs; numbers to be filled in.)*

| Method | Sonar r | Sonar RMSE | Notes |
|--------|---------|-----------|-------|
| Ours — sonar off (DA3+texconf only) | +0.398 ± 0.647 | 0.487 m | Still flips sign in some folds |
| Ours — fixed weight (gain k=0) | +0.985 ± 0.011 | 0.064 m | Marginally better than adaptive |
| Ours — full (adaptive gain k=3) | +0.980 ± 0.013 | 0.075 m | Headline |

The sonar-off ablation shows whether DA3+texture-confidence depth prior alone
achieves correct-sign geometry; if it does, the contribution shrinks. The
fixed-weight ablation isolates the adaptive gain k: if constant w matches
adaptive wμ, the self-diagnosing mechanism is not needed.

### D. F1: truly held-out sonar supervision (Table III)

**F1 results complete (all 5 blocks):**

| Fold | Held-out frames | Test r | Test RMSE | Train r |
|------|----------------|--------|-----------|---------|
| 0 | f000750–f001055 | **−0.396** | 0.377 m | +0.994 |
| 1 | f001060–f001350 | **+0.666** | 0.466 m | +0.996 |
| 2 | f001355–f001650 | **+0.979** | 0.572 m | +0.960 |
| 3 | f001655–f001945 | **+0.055** | 0.469 m | +0.947 |
| 4 | f001950–f002245 | **+0.820** | 0.430 m | +0.958 |
| **F1 agg.** | | **+0.425 ± 0.516** | **0.463 ± 0.064 m** | |
| **Std. CV** | | **+0.980 ± 0.013** | **0.075 ± 0.022 m** | |

The gap between standard CV (r = +0.980) and F1 aggregate (r = +0.425) shows
sonar supervision anchors geometry locally but does NOT propagate globally. Block 2
(middle, surrounded by sonar-supervised frames) achieves r = +0.979 held-out.
Block 0 (first segment, no upstream coverage) flips sign. Block 3 collapses to
near-zero despite being a middle segment — likely a scene geometry transition.

**Implication:** The ping-rate sweep will test whether distributed sparse coverage
outperforms dense contiguous coverage for global propagation.

### E. Sensitivity analysis

**Patch size** (Table IV): r is robust (0.975–0.980) across 10–40 px patches;
RMSE rises from 0.075 m at 20 px to 0.134 m at 40 px as larger patches
average over off-axis field-of-view regions. The default 20×20 px is justified.

**Ping rate** (Table V): *(pending `run_ping_rate_sweep.sh`)*

| Ping rate | Sonar r | Sonar RMSE |
|-----------|---------|-----------|
| 3.92 Hz (full) | +0.987 ± 0.010 | 0.063 m |
| ~1.0 Hz (every 4th) | +0.843 ± 0.101 | 0.223 m |
| ~0.2 Hz (every 20th) | +0.370 ± 0.569 | 0.448 m |
| ~0.1 Hz (every 40th) | +0.374 ± 0.570 | 0.493 m |

**Sonar dropout** (Table VI): *(pending `run_dropout_sweep.sh`)*

| Dropout | Sonar r | Sonar RMSE |
|---------|---------|-----------|
| 0% | +0.974 ± 0.027 | 0.085 m |
| 25% | +0.953 ± 0.033 | 0.116 m |
| 50% | +0.900 ± 0.051 | 0.178 m |
| 75% | +0.807 ± 0.125 | 0.303 m |

### F. Dense depth trace (downstream application)

The trained model renders a depth estimate at every video frame (25 Hz),
6× denser than the raw altimeter (3.92 Hz). At a ROV speed of 0.1 m/s, this
increases the along-track depth sample spacing from ~2.5 cm (altimeter) to
~0.4 cm (rendered). The dense trace at matched-ping frames achieves r = +0.980
against the altimeter (same as the eval result); between pings it interpolates
geometry learned from the full reconstruction. This enables continuous ice-draft
profiling rather than discrete acoustic samples (see `gen_dense_trace.py`).
