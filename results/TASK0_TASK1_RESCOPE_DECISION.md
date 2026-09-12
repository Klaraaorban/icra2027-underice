# Re-scoping decision: Task 0 and Task 1 for single-beam altimeter data

Approved by Klara 2026-08-30 after the multibeam-data audit (no DT101 or any
multibeam data exists in the sample dataset — see the audit findings above this
file in the session transcript; confirmed three independent ways: full channel
header grep across every `.dat` file, the ROV's own `rov.log` instrument-startup
list, and the datalogger's channel-definition XML). Only a Valeport single-beam
altimeter (`valeport.dat`, NMEA `$PRVAT`, one scalar range per ping, no beam
angle or swath) exists.

## Task 0 — re-scoped to 1D along-track spectral test

Original spec compared 2D rasterized surfaces (optical vs. multibeam swath).
Re-scoped version:

1. **Along-track axis**: instead of a 2D grid, use the ROV's actual measured
   1D trajectory. Camera centres from the trained (metric-scaled via the
   calibrated sonar scale) 3DGS model are a real, measured quantity — use
   cumulative horizontal distance between consecutive camera centres,
   projected into the ice-relative frame (plane fit to the altimeter-implied
   ice surface, same idea as the original ice-relative-frame step). This
   avoids relying on GAPS position, which is documented elsewhere in this
   project as sub-survey-grade and degrading under ice.
2. **Two 1D signals on that axis**: (a) the altimeter's own range reading per
   frame (already collected all session, one real measured value per frame),
   (b) the optical reconstruction's rendered center-patch depth per frame
   (same measurement already used throughout this project's sonar diagnostics
   — real, not synthesized).
3. Detrend each 1D signal (subtract a low-order polynomial along-track, same
   intent as the 2D plane+polynomial step), apply a 1D Hann window, compute
   1D FFT power spectral density on a common along-track wavenumber axis.
4. **Cutoff**: two numbers, both reported, conservative (lower) one used as
   the gate, exactly as the original brief intends for the grid-vs-footprint
   comparison:
   - **Sampling-rate Nyquist** (measurable, no missing spec): from median
     along-track spacing between consecutive altimeter pings.
   - **Beam-footprint cutoff** (`d = 2 r tan(half-beam-width)`,
     `k_cut = 1/(2d)`): **BLOCKED** — the Valeport altimeter's beam width is
     not documented anywhere in the dataset, dataset docs, or prior project
     notes. Needs the exact Valeport model number (NMEA talker `$PRVAT` alone
     doesn't identify it) before this can be computed as a real, non-fabricated
     number. Flagged, not guessed. Reported as `k_cut_footprint: MISSING` in
     the verdict output until Klara supplies the model/datasheet.
5. **Split-half coherence control**: unchanged in spirit — train two
   independent 30k reconstructions on alternating-frame halves (same config,
   same seed-count convention as elsewhere in this project), extract the same
   1D along-track optical profile from each, compute magnitude-squared
   coherence gamma^2(k) between the two. Same PASS criterion (gamma^2 > 0.5
   above the claimed band). This part needed no re-scoping — coherence between
   two independently-trained reconstructions is inherently 1D-compatible.

**Caveat that must go in the paper if Task 0 passes**: this is a 1D validation
along the ROV's track, not a 2D areal validation. It proves vision recovers
along-track structure above the altimeter's sampling/footprint limit; it does
NOT by itself prove 2D areal coverage above cutoff (that would need either the
missing multibeam or the Task 5 simulator, which still uses a real synthetic
DT101 and IS run in full 2D).

## Task 1 — re-scoped to time-contiguous holdout

Original spec holds out contiguous *spatial swath blocks* (with ~30% overlap
between adjacent swaths, hence the random-split-leaks warning). With a single
along-track altimeter trace there are no swaths. Re-scoped:

- Hold out **contiguous along-track blocks** (a contiguous span of frames /
  along-track distance), not swaths. This still avoids the same leakage
  problem the original warns about (a random per-frame split would leak,
  since consecutive frames along a slow-moving ROV track are highly spatially
  correlated — the along-track analogue of adjacent-swath overlap).
- This matches the `llffhold`-style split already used throughout this
  project (every 8th frame to test) ONLY IF that split is contiguous-block,
  not interleaved — **audit finding: it is NOT**. The existing convention
  (`llffhold=8`, i.e. every 8th frame by index) is an *interleaved* split, not
  a contiguous block split, and every train/test-split sonar number reported
  earlier in this project (PS117-39's r=0.973, RMSE=0.181m held-out, included)
  used that interleaved convention. Interleaved-every-8th still doesn't share
  exact frames with train, so it is not literally circular, but it is a
  *weaker* holdout than a contiguous block, because immediately-adjacent train
  frames (7 frames away, ~1.4s) are highly spatially correlated with each test
  frame. This should be re-run with genuine contiguous-block holdout for the
  numbers that go in the final paper. Flagged as an action item, not silently
  fixed by rewriting historical numbers.
- `scripts/eval/acoustic_holdout.py` (below) implements the contiguous-block
  version going forward and reports both Pearson r and RMSE in every cell,
  every time, per rule 3/4.

## Task 3 — dropped for real data, kept for simulation

The anisotropic Mahalanobis beam-frame likelihood requires per-ping beam angle
and a real multibeam footprint — not available. Task 3 will be implemented and
tested **only** inside Task 5's HoloOcean simulation, where a synthetic DT101
with the brief's exact specs can be instantiated. No real-data claim will be
made about Task 3 in the paper; simulation-only, stated explicitly.
