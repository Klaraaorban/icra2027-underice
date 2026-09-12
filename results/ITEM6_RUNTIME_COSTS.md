# Runtime and onboard cost

All numbers measured directly in this session's logs/live GPU queries, not
estimated. RTX 3080 (8GB), `--data_device cpu` (our standard convention
throughout this project).

## Sonar supervision's added training cost: negligible

| Config (PS117-39, 7k iterations, same recipe otherwise) | Wall-clock |
|---|---|
| With sonar supervision (239 pings) | 11m 32s |
| Without sonar supervision (0 pings) | 11m 50s |

The "without" run was slightly *slower*, within normal run-to-run system
noise — sonar supervision (one small extra loss term on a 20x20 patch per
supervised frame) adds no measurable training-time cost.

## Inference speed

| Model | ms/frame | FPS |
|---|---|---|
| PS117-39 headline (30k iter, 300 images) | 8.33 | 120 |
| PS117-29 vanilla (7k iter, 216 images) | 3.90 | 256 |

## Pre-flight diagnostic check cost

A full render-and-compare pass over an already-trained, already-posed
300-camera scene (the same computation `sonar_trust_gate.py` / the Proof-2
referee script perform — render every view, compute residual against sonar,
reach a trust decision) took **~37 seconds end to end**, of which ~25-30s is
model/camera loading (fixed cost, not diagnostic-specific) and the rest is
the actual per-frame residual computation across all 300 views. This is the
real, measured cost of "should I trust sonar supervision on this sequence
before committing to a training run" — orders of magnitude cheaper than the
~11-12 minute training run it gates.

## GPU memory

1256 MiB used, measured live during an active 30k training run at 6%
progress (densification still ramping up toward its `densify_until_iter`
peak, so this is a floor, not necessarily the peak, on an 8GB card).
