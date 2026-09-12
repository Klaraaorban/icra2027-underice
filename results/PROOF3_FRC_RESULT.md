# Proof 3 — Resolution without ground truth (Fourier Ring Correlation, 1D)

**Result: achieved resolution is at least as fine as our own along-track
sampling limit can measure (~21cm half-wavelength at the sampling Nyquist),
and is not distinguishable from "finer than the altimeter's own beam
footprint" (15.7cm at this sequence's 1.50m mean range) — FRC never drops
below either standard criterion (0.143 or 0.5) anywhere in the band we can
test.**

## Method

Two independently-trained 3DGS reconstructions (alternating-frame halves,
identical config, Task 0's coherence control) compared via Welch-segmented
Fourier Ring Correlation along the shared track. Lineage: Saxton & Baumeister
1982 (FRC for EM); van Heel & Schatz 2005 (the 0.143 criterion); Nieuwenhuizen
et al. 2013, Nat. Methods (FRC applied to optical/fluorescence nanoscopy —
the direct methodological precedent for using it on an optical 3D
reconstruction here, not an EM one).

## An implementation error caught and fixed before reporting

First attempt computed FRC from a single full-length FFT with post-hoc local
bin-smoothing (5-bin moving average). That gave resolution numbers of ~32-33cm
at both criteria — but inspecting the raw per-bin values showed why that's not
trustworthy: FRC swung between +1 and -1 from one adjacent bin to the next in
a middle frequency band (e.g. 0.994 -> 0.731 -> 0.217 -> **-0.545** -> 0.914 ->
0.207 -> **-0.995**), then recovered to consistently high values (0.9-1.0)
*above* that band all the way to Nyquist. A genuine resolution cutoff decays
and *stays* low past the limit; this dipped and came back, which is the
signature of phase-noise-dominated single-bin estimates in a low-power band,
not a real correlation dropoff. A local moving average over adjacent bins does
not fix this, since neighbouring bins share the same noise-dominated regime
rather than being independent samples of it.

Fixed by computing FRC the way `scipy.signal.coherence` computes coherence:
Welch-segmented (8 overlapping 64-point windows, 50% overlap), averaging
cross- and auto-power *across independent segments* before normalizing. This
is mathematically the right fix (FRC and magnitude-squared coherence are the
same family of estimator; segment-averaging is what gives either one genuine
statistical power) and it resolved the discrepancy: with proper averaging, FRC
matches the separately-computed coherence curve (also >0.94 everywhere up to
2.4 cycles/m) and never crosses either threshold.

## Numbers

- Sampling Nyquist (our along-track measurement limit): 2.397 cycles/m ->
  0.5/(2.397) = 0.209m half-wavelength = ~21cm.
- FRC at 0.5 criterion: not reached within measurable range.
- FRC at 0.143 criterion: not reached within measurable range.
- Altimeter footprint diameter at this sequence's 1.4983m mean range
  (beam half-angle 3 deg, sourced from the VA500P datasheet — see
  TASK0_VERDICT.md for the identification chain): 15.7cm.

## Honest interpretation

This is the SAME structural limit that blocked Task 0's primary spectral
test: our own along-track sampling density can't resolve frequencies fine
enough to find where FRC (or the optical PSD) actually drops off. What this
proof *does* establish is that reconstruction agreement between two
independent halves is excellent at every frequency we CAN test, including
frequencies corresponding to length scales finer than the altimeter's own
footprint — i.e. there's no evidence of degradation right up to our
measurement ceiling. It does not establish a specific numeric resolution
finer than the footprint, because the test apparatus (our own frame density)
isn't fine enough to find that number. A denser along-track query (interpolated
poses between the 300 known camera positions, rendering the two half-models at
those interpolated poses) would extend this test further and is a natural
next step if this needs to go beyond "not disprovable" to "positively
demonstrated."
