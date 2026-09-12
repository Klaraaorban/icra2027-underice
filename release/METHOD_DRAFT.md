# Method section draft (for paper writing)

## III. Method

### A. Scene representation

We use 3D Gaussian Splatting [Kerbl et al., 2023] as our scene representation.
A scene is modelled as a set of 3D Gaussians {G_i}, each with position μ_i,
opacity σ_i, covariance Σ_i, and spherical harmonic colour coefficients c_i.
Rendering proceeds by rasterising the Gaussians in order of depth, accumulating
colour and alpha. We initialise the Gaussians from a depth-aware point cloud
produced by DA-3 [Wang et al., 2024] applied to the training frames, which
provides a dense, scale-consistent depth prior for ice-covered scenes.

### B. Sonar-supervised depth regularisation

Let v denote a camera view, and let d̂(v) be the mean rendered depth over a
20×20-pixel patch centred on the image plane. The Valeport VA500P single-beam
altimeter emits one acoustic pulse per 0.255 s along the ROV's forward axis;
we denote the temporally-nearest ping reading as z_s(v) (in metres), matched
within a 1.0 s tolerance and converted to COLMAP reconstruction units as
z̃_s = z_s / s, where s is a scene-level scale factor.

The sonar supervision term is:

    L_sonar = w · μ(v) · |d̂(v) − z̃_s(v)|

where w = 0.1 is a base weight and μ is an adaptive multiplier:

    μ(v) = clip(1 + k · |d̂(v) − z̃_s(v)|_sg, 0.5, 4.0)

The subscript _sg denotes stop-gradient: μ is computed from the detached
residual so it modulates the loss weight without contributing its own gradient.
With gain k = 3.0, frames where the rendered depth deviates by 0.1 m from the
sonar ping receive weight w · 1.3, and those deviating by 0.2 m receive w · 1.6,
up to a ceiling of 4w at deviations ≥ 1/k ≈ 0.33 m. The floor (0.5w) prevents
near-zero sonar loss from being dominated by gradient noise at well-converged
frames.

This self-diagnosing weight is the inverse of the usual robust-loss philosophy
(which downweights large residuals as outliers): here the altimeter is more
reliable than the renderer during early training, so frames with the largest
depth errors are the ones most in need of correction.

### C. Scale fitting

The scale factor s converts the COLMAP reconstruction's arbitrary depth units
to metres. We fit s once on the training split of the full-dataset model via:

    s* = Σ_train z_s · d̂ / Σ_train d̂²

(least-squares regression of sonar on rendered depth, no intercept). This
fitted s* = 5.24 for PS117-39, and is held fixed for all ablation variants
(including the no-sonar model, which would otherwise have free choice of s).

### D. Depth initialisation and texture weighting

The training loss combines the standard 3DGS photometric objective with L_sonar:

    L = L_photo + L_sonar + λ_d · L_depth

where L_depth is an L1 penalty on rendered depth against the per-pixel DA-3
depth prior. The DA-3 depths are modulated by a texture-confidence mask
τ ∈ [0, 1] estimated from local gradient magnitude and colour variance:
high-texture regions (τ → 1) receive stronger depth regularisation than
low-texture ice faces (τ → 0), preventing the depth prior from forcing flat,
low-confidence areas into an incorrect configuration.

### E. Evaluation protocol

We report Pearson r and RMSE against held-out sonar pings using a 5-fold
contiguous along-track block holdout. Adjacent frames on a slow ROV are
highly autocorrelated; contiguous block holdout is the hardest and most
realistic split. For each fold, the scale factor is fitted on the training
frames only, then fixed for the test frames.

For the F1 experiment (truly held-out sonar supervision), we train five
separate models — one per fold — with the corresponding frames removed from
L_sonar during training, not just from scale fitting. This tests whether the
sonar constraint generalises across the scene, not just whether it memorises
the training pings.

### F. Baselines

**SeaSplat [Yang et al., 2024]:** 3DGS extended with an underwater appearance
model (absorption + backscatter per Gaussian). Trained at 7k and 30k iterations;
we report both. Evaluated with `eval_seasplat.py` (same 5-fold CV).

**WaterSplatting [Huang et al., 2024]:** 3DGS with a volumetric water column.
Evaluated with `eval_watersplatting.py` (same 5-fold CV).

Neither baseline has access to sonar data. Both are evaluated on the same PS117-39
dataset with identical held-out splits.
