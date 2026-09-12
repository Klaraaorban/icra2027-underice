# Methods supplement — sonar-supervised 3DGS

Technical details for reviewers and reproducers. Companion to `Sonar_Story.pdf`.

## Self-diagnosing sonar-depth regularization loss

The loss is applied at every training iteration for each camera view whose
frame stem appears in `sonar_depths.json`. Let `d̂(v)` be the mean rendered
depth over a 20×20 px patch centred on the image (matching the altimeter's
forward-looking beam), and let `z_s` be the sonar reading for that view
converted to COLMAP units:

```
z_s  =  sonar_depth_metres / sonar_scale

L_sonar  =  w · μ · |d̂(v) − z_s|

where:
  μ = clip(1 + k · |d̂(v) − z_s|.detach(), 0.5, 4.0)   [adaptive multiplier]
  w = sonar_loss_weight                                   [base weight, 0.1]
  k = sonar_adaptive_gain                                 [adaptive gain, 3.0]
```

The multiplier μ is computed from the detached residual so it modulates the
loss weight but does not itself propagate gradients — the gradient path is
purely through `|d̂(v) − z_s|`. Setting `k = 0` (adaptive gain off) gives
μ = 1 and reduces to a plain L1 depth loss with fixed weight.

**Which frames see the sonar loss:** only views whose frame stem is present in
`sonar_depths.json` at training time. All 298 frames have a sonar reading in
the full-dataset training. In the F1 block-holdout experiments,
`sonar_depths_holdout_block{k}.json` removes one temporal block's frames
before training, so those frames receive zero sonar loss during that run.

**sonar_scale:** 5.2421 — fit once on the train split of the full-dataset
model via least-squares `scale = Σ(z_sonar · d̂) / Σ(d̂²)`, stored in
`sonar_scale.json`. The same value is reused for all ablation runs to avoid
confounding the comparison.

## Sonar–camera alignment (boresight assumption)

The Valeport VA500P sends a single acoustic beam along the instrument's
forward axis. On the ROV used in PS117, the altimeter and camera share the
same forward-facing mount, so the beam nominally points at the same target as
the optical axis. **No formal extrinsic calibration was performed.** The
evaluation uses a 20×20 px centre patch to average over small boresight
misalignment, and the correlation metric (Pearson r) is insensitive to a
fixed angular offset as long as it does not change sign over the sequence.

If a detailed extrinsic calibration exists in the cruise documentation, the
patch size (`--patch-size` arg) can be replaced by a projected beam footprint.
A boresight error that introduces a depth bias would raise RMSE but leave r
approximately unchanged; a sign-changing deflection would be visible as
reduced r at specific range-to-ice values.

## Convergence and training budget

At 30,000 iterations both photometric metrics (PSNR: 33.86 → 36.44 dB) and
the independent held-out sonar RMSE (0.244 → 0.075 m) improve substantially
versus 7,000 iterations, with no signs of overfitting. SeaSplat was trained to
only 7,000 iterations in the original evaluation (PSNR 21.75 dB, which reads
as undertrained). The matched-budget ablation at 30,000 iterations (script:
`run_ablations.sh`, result: `results_ablation_seasplat_30k.json`) closes this
comparison gap.

## Computational requirements

Hardware: NVIDIA RTX 3080 Laptop GPU, 8 GB VRAM.

| Stage | Time |
|-------|------|
| COLMAP reconstruction (298 frames) | ~15 min (CPU, standard pipeline) |
| DA3 depth estimation (298 frames) | ~8 min (GPU) |
| 3DGS training at 30k iterations | ~55-90 min (GPU) |
| Sonar eval (5-fold, rendering 298 frames) | ~3 min (GPU) |

Data storage: ~2.1 GB per model checkpoint (point cloud + training state at
30k iterations). Five F1 holdout models require ~10.5 GB additional disk.

Peak GPU memory during training: 8 GB (near the device limit; `--data_device cpu`
is required to offload the image cache).

## Sign inversion and SfM initialisation quality

The sign inversion in baselines is not explained by poor SfM initialisation.
On PS117-29 (harder station, 72% registration rate), MASt3R-SfM produces a
right-signed initial geometry (Pearson r = +0.067 against sonar on the train
split) — correctly indicating that the ROV is moving toward the ice. Pure
photometric 3DGS training on top of that initialisation corrupts it to
r = −0.31: the optimizer finds a locally photometrically consistent solution
with the depth direction reversed, because on low-texture ice the rendering
loss is nearly flat in depth. This shows the sign inversion is a property of
the photometric training objective on this scene type, not a property of any
particular SfM initialisation algorithm.

**PS117-29 baseline (5-fold CV at 30k iterations):** test r = −0.197 ± 0.421,
RMSE = 1.615 ± 0.570 m. The high variance across folds (some blocks flip to
slightly positive r) reflects that without sonar supervision the model is
near-random w.r.t. depth, not locked into a single sign like the COLMAP-init
baselines. The qualitative conclusion is the same: purely photometric training
does not produce a reliable sonar-consistent geometry at this station either.
Sonar-supervised results on PS117-29 are a planned follow-on experiment
(requires training a separate model there; data and scripts available).

## Patch-size sensitivity

Pearson r is robust to the patch size choice: sweeping the centre patch from
10×10 to 40×40 px gives r = +0.977 / +0.980 / +0.975 respectively (std ≈ 0.013–0.016
across folds). RMSE rises from 0.075 m at 20×20 to 0.134 m at 40×40 because a
larger patch averages over field-of-view regions that are increasingly far from
the single-point altimeter beam, so the default 20×20 px is the best trade-off.

## Evaluation methodology choices

| Choice | Rationale |
|--------|-----------|
| 5-fold contiguous block holdout | Adjacent frames on a slow ROV are highly correlated; interleaved every-Nth splits leak this correlation into the test metric. A contiguous block is the hardest and most realistic split. |
| Scale fit on train split only | Scale is a free parameter; fitting it on the test split would let a sign-inverted model achieve arbitrary RMSE. Fit once on train, apply to test. |
| Pearson r as headline metric | Scale-invariant; unaffected by the one fitted scale parameter. Reports whether the depth signal is informative and correctly oriented. RMSE quantifies the residual after the scale parameter is removed. |
| 20×20 px centre patch | Best trade-off in the patch-size sweep (see above). Robust to small boresight misalignment; larger patches raise RMSE by including off-axis pixels. |
