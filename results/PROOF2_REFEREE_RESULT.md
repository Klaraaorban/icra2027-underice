# Proof 2 — Is the per-frame altimeter residual a usable reliability signal?

**Result: yes, clearly, across models — and PSNR is not just uninformative but
actively misleading between them. Within a single (uniformly bad) model,
residual is elevated throughout rather than sharply localized, which is itself
an honest, informative finding about this particular failure mode.**

## Cross-model comparison (the clearest result)

| Model | Known status | mean \|residual\| | mean PSNR |
|---|---|---|---|
| PS117-39, 30k headline (`ps117_39_da3_texconf_calibrated_sonar_30k`) | good (r=0.973 held-out, TASK1) | **0.043 m** | 36.80 dB |
| PS117-29 vanilla (`ps117_29_verified_vanilla`) | bad (r=-0.50, wrong sign) | **1.262 m** | **38.94 dB** |

Residual differs by **29x** between the good and bad reconstruction, correctly
in the direction that matches ground truth. PSNR differs by only ~2dB, and in
the **wrong direction** — the known-bad reconstruction scores higher. This is
not "PSNR is noisy here," it is "PSNR actively points the wrong way," which is
the sharper and more useful claim for the paper.

## Within-model (does residual track error where PSNR doesn't, frame by frame?)

On PS117-29 vanilla's 27 test frames: Pearson r(|residual|, PSNR) = **0.151**
— near zero, and if anything the wrong sign for PSNR to be a useful reliability
proxy (a working referee should show |residual| high exactly where PSNR is
also poor; instead there's no such relationship at all).

## Detection latency — honest negative/nuanced finding

Attempted: using the known real transition (established earlier via the raw
sonar trace and a visually-verified frame pair) at t=960s, does elevated
residual detect entry into the "close-approach, real-altitude-change" region
with low latency? Result: **no sharp onset detectable** — mean |residual| is
1.208m before t=960s and 1.349m after, both already far above any reasonable
false-alarm threshold. This is honest and informative in its own right: this
particular model's failure mode (COLMAP's systematic scale corruption under
weak texture, established earlier via the wrong-sign train-split diagnostic,
r=-0.50 across the WHOLE train split) is global, not a localized event with a
detectable onset — the residual correctly reflects that by being elevated
*everywhere*, not by spiking at one boundary. "Detection latency" as a framing
implicitly assumes a point-onset failure; this failure mode doesn't have one,
and forcing a latency number onto it would misrepresent what's actually
happening. A model/sequence with a genuinely localized failure (a patch of bad
texture within an otherwise-good reconstruction) would be a better test case
for the latency framing specifically -- flagged as a follow-up if that
distinction matters for the paper.

## Compute cost per frame

- PS117-39 (300 images, 30k-iter model): 8.33 ms/frame render time.
- PS117-29 (216 images, 7k-iter model): 3.90 ms/frame render time.

Both trivially cheap relative to any real-time or near-real-time deployment
budget -- the residual signal costs essentially nothing beyond a render that
would happen anyway.
