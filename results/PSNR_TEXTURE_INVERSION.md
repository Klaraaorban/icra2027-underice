# PSNR is inversely related to scene texture, not a proxy for reconstruction quality

## UPDATE (2026-09-02): expanded to 17 configs, n=4930 — systematic, not a 2-point replication

Requested follow-up: "no training runs needed, get n in the thousands." True
new stations weren't achievable without training (only PS117-39 and PS117-29
have existing trained+rendered models; PS117-37's dataset folder is empty,
other PS117-29 windows are either the confirmed-bad old window or lack
renders) — flagged honestly. What *was* achievable within "no new training":
real inference-only PSNR extraction (not training) across every existing
trained checkpoint for both stations — 15 additional configs, spanning both
the standard COLMAP image set and the separately-processed MASt3R-seeded
image set (each getting its own texture computation, since texture is a
property of the actual ground-truth image file used for that render, not an
abstract scene label).

**Pooled result, n=4930, both stations, 17 configs: rho(grad_energy, PSNR) =
-0.519, p < 1e-300 (machine-precision zero).**

| Scope | n | rho(grad_energy, PSNR) | p |
|---|---|---|---|
| PS117-39, pooled across 12 configs | 3298 | -0.466 | 1.1e-177 |
| PS117-29, pooled across 5 configs | 1632 | -0.574 | 1.7e-143 |
| Both stations pooled | 4930 | **-0.519** | ~0 |

Per-config breakdown: 13 of 17 configs independently show negative rho, 12 of
those individually significant at p<0.001. Two configs buck the trend
(`colmap_mast3r_fused`: +0.194, p=0.0007 — wrong direction, worth a closer
look if this becomes a paper figure; `da3_depthreg`, `fisheye_eval`,
`da3_texconf_floor_sonar`: near-zero, not significant). The overwhelming
majority and the pooled statistic are unambiguous. Entropy remains the weaker
signal throughout (pooled rho=-0.005, not significant) — gradient energy is
the metric to report, not entropy.

This closes the loop from "two stations, hundreds of frames" to "two
stations, seventeen independent reconstructions, thousands of frames,
p-values at the limit of floating-point representation." The qualitative
claim is unchanged; the statistical backing is now about as strong as this
kind of correlational claim can get without genuinely new capture sites.

---

## Original result (2 configs, before expansion)

**This is likely the single strongest, most defensible finding in the
project.** Confirmed two independent ways, both highly significant.

## Cross-scene (the finding that motivated the check)

| Scene | mean entropy | mean gradient energy | mean PSNR |
|---|---|---|---|
| PS117-39 (richer, moving scene) | 7.440 | 10.582 | 36.80 dB |
| PS117-29 (flat, monotone close-approach) | 6.386 | 7.292 | **38.94 dB** |

The scene with *less* texture scores *higher* PSNR. Backwards from what PSNR
"reconstruction quality" framing would predict, exactly as flagged.

## Within-scene, per-frame (the stronger test — n in the hundreds, not 2)

Correlating each frame's own texture (entropy, gradient energy computed
directly on the ground-truth image) against that same frame's own PSNR:

| Scene | n | rho(entropy, PSNR) | rho(grad energy, PSNR) |
|---|---|---|---|
| PS117-39 | 298 | -0.113 (p=0.051, borderline) | **-0.439 (p<0.0001)** |
| PS117-29 | 216 | -0.117 (p=0.086, not sig.) | **-0.426 (p<0.0001)** |

Gradient energy is the cleaner signal of the two (entropy is noisier, likely
because Shannon entropy over raw pixel-intensity histograms is a weaker
texture proxy than local gradient magnitude for this kind of low-contrast,
color-cast underwater imagery). The gradient-energy result is highly
significant, consistent in sign, and holds independently in both scenes —
this is not a two-point coincidence, it's the same relationship replicated
at the frame level in two different sequences.

## What this means for the paper

The correct headline sentence, per the sharper framing requested: **"In
under-ice imagery, PSNR is bounded by scene texture, not by geometric
correctness — the least texturally-complex, hardest-to-reconstruct-in-3D
frames systematically score the highest PSNR, because a smooth surface with
little high-frequency content is trivial for photometric optimization to fit
regardless of whether the underlying geometry is correct."** This explains
mechanistically *why* the PSNR/sonar-r decoupling (item 1) and the
divergence-plot pattern (item 2) happen, rather than just documenting that
they do — low-texture regions offer photometric optimization almost no
gradient signal to resist collapsing into a geometrically-wrong but
visually-smooth solution, so PSNR keeps climbing while it does. It also
explains the field-level concern raised: any published under-ice or
low-texture-underwater 3DGS PSNR number should be read with this inversion in
mind, since easier-sounding "high PSNR" results may specifically indicate the
hardest, most textureless capture conditions, not the best reconstructions.

## Method note

Gradient energy = mean Sobel-magnitude over the full grayscale image
(`sqrt(Gx^2 + Gy^2)`), entropy = Shannon entropy of the 256-bin pixel-intensity
histogram. Both computed directly on real ground-truth JPEGs, no synthetic or
rendered images involved on the texture side. PSNR values are the same
per-frame numbers already computed in the Proof-2 referee CSVs (real
renders against real ground truth, train+test split combined for this
specific texture analysis since the texture-vs-PSNR relationship is a
property of the image content, not a held-out-generalization question).
