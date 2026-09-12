# Sonar-accuracy evaluation

Scores a 3D reconstruction's geometry against a real single-beam altimeter
(sonar) trace: fit a scale factor on a train split, then report Pearson r
and RMSE on a held-out test split. Same methodology across all three
scripts so the numbers are comparable.

All results use 5-fold contiguous along-track block holdout (mean ± std).

| Method | PSNR | Sonar r | Sonar RMSE |
|---|---|---|---|
| Ours (adaptive sonar, 30k) | 36.44 dB | +0.980 ± 0.013 | 0.075 ± 0.022 m |
| SeaSplat (no sonar, 7k) | 21.75 dB | −0.234 ± 0.771 | 1.460 ± 0.397 m |
| WaterSplatting (no sonar) | 30.24 dB | −0.245 ± 0.754 | 1.416 ± 0.387 m |

The two baselines have no sonar supervision and no way to anchor absolute
(or even correctly-signed) depth on their own — both come out
anti-correlated with real range, not just weak. The high variance (±0.77)
reflects spatially inconsistent geometry, not measurement noise.

## Ablation results (in `results_ablation*.json` when available)

The scripts `run_ablations.sh` and `run_f1_holdout_training.sh` in the
gaussian-splatting repo produce the following additional comparisons:

| Method | Sonar r | Sonar RMSE | Notes |
|---|---|---|---|
| Ours — F1 truly held-out | (pending) | (pending) | Each model trained without sonar on its test block |
| Ours — fixed weight (gain=0) | (pending) | (pending) | Sonar on, no adaptive gain |
| Ours — no sonar | (pending) | (pending) | DA3+texconf only, sonar off |
| SeaSplat 30k | (pending) | (pending) | Matched training budget |

## PS117-29 multi-station results

A second deployment, PS117-29 (harder station, 72% COLMAP registration rate,
MASt3R-SfM initialisation). 300 frames, 216 matched sonar readings.

| Model | Sonar r | Sonar RMSE | Notes |
|---|---|---|---|
| Baseline (MASt3R-seeded, no sonar) | −0.197 ± 0.421 | 1.615 ± 0.570 m | `results_ps29_baseline.json` |
| Ours (sonar-supervised, 30k) | (planned) | (planned) | train script: `run_ps29_sonar.sh` |

The high block-to-block variance in the baseline (r ranges from −0.726 to +0.446
across folds) shows the photometric optimizer finds locally consistent but
globally un-anchored geometry. Ours on PS117-29 will use identical hyperparams
to PS117-39 (`sonar_loss_weight=0.1`, `sonar_adaptive_gain=3.0`, 30k iter).

Patch-size sensitivity (PS117-39 30k model):

| Patch | r | RMSE |
|-------|---|------|
| 10×10 | +0.977 ± 0.013 | 0.085 ± 0.021 m |
| 20×20 | +0.980 ± 0.013 | 0.075 ± 0.022 m |
| 40×40 | +0.975 ± 0.016 | 0.134 ± 0.044 m |

r is robust; 20×20 gives the best RMSE.

## Files

- `sonar_utils.py` -- altimeter loading/matching, scale-fit-and-score. Used
  by all three eval scripts.
- `eval_ours.py` -- our sonar-supervised model (`gaussian-splatting` repo).
  Supports `index_to_orig_name` config field for datasets with sequential
  image indices (e.g. PS117-29).
- `eval_seasplat.py` -- SeaSplat checkpoint, same 5-fold contiguous-block CV.
- `eval_watersplatting.py` -- WaterSplatting (nerfstudio) checkpoint, same CV.
- `configs/ps117_39.yaml` -- PS117-39 sensor/timing config; fill in your own paths.
- `configs/ps117_29.yaml` -- PS117-29 config template (harder station, partial COLMAP).
- `configs/ps117_29_mast3r.yaml` -- PS117-29 config with MASt3R-seeded dataset.
- `test_sonar_utils.py` -- unit tests for `sonar_utils.py` (no GPU needed).
- `results_ours.json` -- 5-fold CV results for our model (reproducible without retraining).
- `results_ours_patch{10,20,40}.json` -- patch-size sensitivity sweep results.
- `results_seasplat.json`, `results_watersplatting.json` -- baseline results.
- `results_ours_f1.json` -- F1 truly-held-out results (available after `run_f1_holdout_training.sh`).
- `results_ablation*.json` -- ablation comparisons (available after `run_ablations.sh`).
- `results_ps29_baseline.json` -- PS117-29 baseline (no sonar, MASt3R-seeded).
- `results_ps29_ours.json` -- PS117-29 sonar-supervised (available after `run_ps29_sonar.sh`).

## Requirements

Each script needs the conda env of the model it evaluates. Exported
environment files are in `envs/`:

| Script | Env file | Upstream repo | Commit |
|--------|----------|---------------|--------|
| `eval_ours.py` | `envs/gaussian_splatting.yml` | github.com/graphdeco-inria/gaussian-splatting | `54c035f` |
| `eval_seasplat.py` | `envs/seasplat_py310.yml` | github.com/dxyang/seasplat | `ddc6259` |
| `eval_watersplatting.py` | `envs/gsplat_env.yml` | github.com/water-splatting/water-splatting | `0c2d943` |

Restore an env with: `conda env create -f envs/<file>.yml`

Additional notes:
- `eval_seasplat.py` must be run **from inside the SeaSplat repo root** (its
  `scene`/`gaussian_renderer`/`arguments` modules share names with the main
  gaussian-splatting repo, so cwd matters).
- `eval_watersplatting.py`: on Windows set `PYTHONUTF8=1` before running
  (nerfstudio prints a Unicode character that crashes the default codepage).

## Sonar log format

One JSON object per line:

```json
{"timestamp": "2019-01-10T12:44:30.123456", "Ping Depth": 1.482}
```

## Metrics

Each script reports:

- **Pearson r** — scale-invariant; measures whether the model's depth
  and real sonar range move in the same direction and shape. The headline
  sign-inversion finding lives here.
- **RMSE** — post-scale-fit residual; measures how well the shape matches
  in metric units after the one free scale parameter is applied.
- **RMSE (unscaled)** — RMSE at s=1, i.e. treating the raw rendered depth
  as if it were directly in metres. If this is close to the scaled RMSE,
  the model is already metric-accurate; a large difference means the model
  is geometrically right but scale-wrong.

All three metrics are the primary quantitative claim. PSNR, SSIM and LPIPS
measure photometric quality only, not geometry; the 36.44 / 21.75 / 30.24 dB
figures in `Sonar_Story.pdf` come from the respective training-framework log
files (gaussian-splatting train.py stdout, WaterSplatting nerfstudio eval),
not from any script in this repo.

## Note on depth units across methods

Each method's raw rendered depth is in its own internal units (not
necessarily metres). A single scale factor is fit on the train split to map
each method's depth into sonar metres for RMSE computation. Pearson r is
unaffected by this. All three scripts apply the same fitting procedure so
the comparison is fair, but the scale factors are model-specific and should
not be compared directly between methods.

## Note on the held-out split

All three scripts now use the same 5-fold contiguous along-track block
holdout. Adjacent frames on a slow-moving ROV are highly correlated, so a
contiguous block is a harder and fairer test than interleaved every-Nth-frame
splits, which leak correlation between train and test.
