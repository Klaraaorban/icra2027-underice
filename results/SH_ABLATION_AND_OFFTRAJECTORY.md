# SH ablation and off-trajectory degradation — both tested, neither confirms the predicted clean mechanism

Both real results, reported as found rather than fit to the hypothesis.

## SH ablation (degree 3 vs degree 0)

Same-iteration (7k), same-methodology (contiguous-block holdout) comparison
on PS117-39:

| Config | PSNR (test) | sonar r (held-out) | sonar RMSE |
|---|---|---|---|
| Full SH (degree 3) | 33.84 dB | 0.907 ± 0.026 | 0.180 ± 0.040 m |
| SH degree 0 | 34.38 dB | **0.928 ± 0.050** | **0.150 ± 0.033 m** |

**Sonar r did improve as predicted.** But PSNR did not drop — it also
improved slightly. This is not the clean tradeoff the mechanism hypothesis
predicted (view-dependent color absorbing an illumination confound at
geometry's expense); instead both metrics moved the same direction. Honest
reading: reducing SH degree may simply make optimization easier at this
iteration budget (fewer parameters to fit), which helps both objectives
together, rather than specifically freeing geometry from an illumination
confound. A real test of the confound-specific mechanism would need to hold
photometric fit ability roughly constant (e.g. compare at matched final loss,
or at a much later iteration where full-SH has had time to actually exploit
its extra freedom to overfit color at geometry's expense) rather than at a
fixed iteration count where the simpler model just optimizes faster. Flagged
as the natural follow-up if this line is worth pursuing further; the result
as measured does not support the causal-mechanism claim as strongly as hoped,
though it is directionally consistent with "less view-dependent freedom
correlates with better geometry."

## Off-trajectory degradation

Real held-out frames' 3D distance to nearest training camera vs. that
frame's own PSNR (not synthetic poses, which would lack ground truth) —
PS117-39, sonar-supervised vs. non-sonar twin, both at the same 7k iteration
and contiguous-block split:

| Model | n | rho(distance, PSNR) | p |
|---|---|---|---|
| Sonar-supervised | 38 | -0.331 | 0.043 |
| Non-sonar twin | 38 | -0.320 | 0.050 |

**Both models degrade with distance, at essentially the same rate.** No
meaningful difference between conditions in the real-data-available distance
range (0.3-1.1m). This does not confirm "vanilla falls off a cliff,
sonar-supervised decays gently."

Extended to a larger, synthetic 1.5m + 35 degree offset (matched between the
two models, same reference frame) to check whether the effect only appears at
larger offsets than real held-out data covers: the two renders are visually
near-identical (both show the same coverage void at the frame edge and
similarly soft, blob-like ice structure) -- see
`results/figures/offtraj_sonar_1p5m/` and `offtraj_nosonar_1p5m/`.

**Honest conclusion: at this configuration and these offset scales, sonar
supervision does not measurably change off-trajectory robustness.** This
does not support using off-trajectory PSNR as a sonar-free circularity
defence for this particular recipe. It may still be true for a different
comparison (e.g. a MUCH larger offset scale, or comparing against a model
with dramatically worse initial geometry like the PS117-29 vanilla case in
item 5's figure, where the effect was visually obvious) -- the item-5 figure's
dramatic degradation was on a model already known to have badly wrong
geometry (wrong-sign scale), not on a matched sonar-vs-non-sonar pair with
otherwise-similar geometry quality. The distinction that emerges: off-trajectory
degradation tracks OVERALL geometric correctness, not specifically whether
sonar supervision was used -- consistent with, not contradicting, the
project's broader finding that geometric correctness and photometric fit are
decoupled, but not the specific causal story hypothesized here.
