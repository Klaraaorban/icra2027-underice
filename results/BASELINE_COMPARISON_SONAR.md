# Baseline comparison, PS117-39 — photometric + sonar geometry, side by side

**The missing baseline sonar numbers are filled in.** Neither SeaSplat nor
WaterSplatting has ever seen the sonar (no supervision, no calibration to
it) — this evaluates each one's raw geometry against real sonar the same way
our own pipeline is scored, using each baseline's own already-trained
checkpoint (no retraining).

| Method | PSNR | SSIM | LPIPS | Sonar r (held-out) | Sonar RMSE (held-out) |
|---|---|---|---|---|---|
| **Ours (self-diagnosing sonar-supervised, 30k)** | **36.44 dB** | **0.937** | **0.316** | **+0.973** | **0.181 m** |
| SeaSplat (no sonar) | 21.75 dB | 0.844 | 0.388 | -0.861 | 1.340 m |
| WaterSplatting (no sonar) | 30.24 dB | 0.926 | 0.423 | -0.859 | 1.274 m |

## The point

It's not just that our method wins. **Both baselines are wrong-signed** —
not weakly correlated with real depth, *anti*-correlated (r ≈ -0.86, both
of them, independently, two completely different codebases). RMSE sits
around 1.3 m for both — roughly **7x worse** than our sonar-supervised
result — and that RMSE number is almost beside the point when the sign
itself is backwards: as real range increases, both baselines' estimated
depth systematically decreases.

Two independent underwater-physics 3DGS/NeRF methods, with two different
SfM initializations and two different volumetric/scattering models, land on
the *same* failure — that's evidence the sign inversion isn't a
scene-specific fluke or an implementation bug in one codebase, it's a
structural gap: nothing in either pipeline anchors absolute (or even
correctly-signed relative) depth without an external metric reference. A
single cheap single-beam altimeter is exactly that reference. This is the
strongest form of the paper's argument: it's not "our sonar trick helps a
little," it's "state-of-the-art underwater 3DGS/NeRF baselines get the
*direction* of depth wrong on this real polar footage, and a $-cheap sensor
most ROVs already carry fixes it."

Also worth noting: **our method's PSNR is the highest of the three** despite
carrying the extra sonar supervision term — the two aren't in tension here.

## Caveat, disclosed rather than buried

The three rows use three different held-out splits, each the *native*
convention of its own framework (not re-split, since none of this required
retraining):

- Ours: contiguous along-track block holdout (this project's own, most
  rigorous protocol — designed specifically to prevent adjacent-frame
  leakage).
- SeaSplat: its `eval=True` interleaved split (same convention this project
  moved away from for its own headline numbers).
- WaterSplatting: nerfstudio's `ColmapDataParser` `eval_mode=interval`,
  `train_split_fraction=0.9` (roughly every 10th frame).

None of these are contiguous-block holdouts, so they're all somewhat weaker
splits than ours (nearby train frames are highly correlated with each
interleaved test frame). If this table goes in the paper, note the
per-method split explicitly rather than presenting them as identically
strict — though given the baselines are catastrophically *sign-inverted*,
not just weaker, a stricter holdout would not be expected to change the
qualitative conclusion, only possibly the exact RMSE.

## Sources

- Ours: `results/runs/ps117_39_30k_acoustic_holdout.json` (PSNR/SSIM/LPIPS
  from the corresponding training run's own eval log, matching the r=0.973,
  RMSE=0.181m headline number reported in project memory)
- SeaSplat: `results/runs/seasplat_ps117_39_sonar_eval.json`, PSNR/SSIM/LPIPS
  from the checkpoint's own `eval_metrics.json`; sonar eval via
  `scripts/seasplat_sonar_eval.py`
- WaterSplatting: `results/runs/watersplatting_ps117_39_sonar_eval.json`,
  PSNR/SSIM/LPIPS from `results/runs/watersplatting_ps117_39_eval_metrics.json`
  (re-run via `ns-eval` this session, not just recalled from memory); sonar
  eval via `scripts/watersplatting_sonar_eval.py`
