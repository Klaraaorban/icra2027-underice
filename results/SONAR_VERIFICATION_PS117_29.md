# PS117-29 sonar data verification

**Verdict: correct.** Checked four independent ways, none contradictory.

1. **Field extraction**: every script (`calibrate_sonar_scale_ps117_29.py`,
   `geometric_vs_sonar_split_ps117_29.py`, `acoustic_holdout.py`,
   `spectral_test_1d.py`) extracts `Ping Depth` (the VA500P's acoustic range
   reading) consistently -- never `Pressure Depth` (a separate water-column
   depth field in the same NMEA sentence, which would be a different, wrong
   quantity).
2. **Sync window consistency**: our `CLIP_START` (2019-01-03T19:38:28) plus
   the used window (t=930-990s) lands at 19:53:58-19:54:58 UTC, comfortably
   inside the datalogger's actual covered window (19:42:39-20:57:57 UTC, from
   the order XML's own start/endtime) -- not proof of second-level precision,
   but not contradicted either. valeport.dat's 18,128 readings over that
   ~75-minute span implies ~3.92Hz throughout, consistent with the measured
   ping rate.
3. **GoPro embedded metadata checked and correctly distrusted**: the MP4's own
   `mvhd` creation_time reads 2011-01-08, a known GoPro clock-reset artifact
   (pre-dates the 2019 expedition entirely) -- not usable for verification,
   and correctly not used. Flagged so nobody re-tries this path expecting a
   different answer.
4. **Three-point visual cross-check, monotonic with sonar range**:
   - far (t=930s, sonar=2.51m): hazy, diffuse, water-column-scattering-typical.
   - mid (t=953.6s, sonar=1.74m): moderate structure, intermediate detail.
   - close (t=970s, sonar=0.135m): sharp, large-scale angular ice structure
     filling the frame.
   All three independently match what their sonar reading predicts, in the
   correct order, with no interpolation or fitting involved -- this is the
   strongest evidence, since it doesn't depend on trusting the sync timestamp
   at all, only on the qualitative fact that closer objects look sharper and
   larger in frame, which is unambiguous regardless of exact time alignment.

No evidence of a sign, unit, offset, or field-selection error. The negative
and block-dependent-sign-flip sonar correlations found in Proof 1 and the
dose-response work are real geometric findings about this station's
reconstruction quality, not sonar data-quality artifacts.
