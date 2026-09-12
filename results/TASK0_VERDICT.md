# Task 0 Verdict — PS117-39 (easy tier), real single-beam altimeter data

**VERDICT: STRUCTURALLY INCONCLUSIVE on real data — not a processing failure, a
sensor-physics limitation. Full reasoning chain below. Claim C's real-data proof
routes through Task 5 (simulator) instead; see recommendation at the end.**

## Chain of findings, in order

1. No multibeam exists in this dataset (established earlier, three independent
   ways: channel-header grep, `rov.log` instrument list, datalogger XML).
2. Re-scoped to a 1D along-track spectral test using the real Valeport
   altimeter trace (`scripts/spectral_test_1d.py`) — every point measured, no
   interpolation across gaps, raw per-ping soundings (not a gridded product).
3. First run: no beam-width spec available, so the only computable cutoff was
   the along-track sampling Nyquist (2.397 cycles/m, from 0.2086m median
   camera-frame spacing) — by construction this equals the FFT's own Nyquist
   limit, so there is no frequency band *above* cutoff to test. Correctly
   returned INCONCLUSIVE rather than a fabricated verdict.
4. Traced the actual sensor: the PS117 cruise report's own ROV-sampling
   section (SIPES 2 project — co-authored by M. Milnes, the same person named
   in this dataset's own `README.txt`) describes this specific ROV's payload
   as hyperspectral radiometers + video + stills camera, matching this
   dataset's file inventory exactly (`radiance.dat`, `irradiance.dat`,
   `gopros/`, `upcam/`). No multibeam is mentioned. This is a *different*,
   simpler ROV deployment than the Imagenex DT101-equipped "V8 M500" platform
   described in Katlein et al. 2017 (Frontiers Mar. Sci. 4:281), which is tied
   in its own text to the 2019/20 MOSAiC expedition specifically, a later,
   separate deployment. Also corrects a beam-count detail: that platform paper
   states 420 beams for the DT101, not 480 — moot for our data either way,
   since it isn't the instrument used here.
   Source: [PS117 Expedition Programme (BODC/AWI cruise report)](https://www.bodc.ac.uk/resources/inventories/cruise_inventory/reports/polarstern_ps117.pdf),
   ROV sampling section (SIPES 2, p.31).
5. Identified the actual altimeter with high (not absolute) confidence: a
   **Valeport VA500P** — the NMEA sentence's paired "Ping Depth" (acoustic)
   + "Pressure Depth" (integrated pressure sensor) fields match the VA500P's
   documented optional integrated pressure sensor exactly. Manufacturer
   datasheet: **Beam Angle ±3°**, 500kHz, 0.1-100m range, 1/2/4Hz or continuous
   data rate. Source:
   [Valeport VA500 Altimeter Datasheet](https://www.valeport.co.uk/content/uploads/2020/04/VA500-Altimeter-Datasheet-April-2020.pdf).
6. Re-ran with the real, sourced beam width. Result at PS117-39's actual
   mean survey range (1.50m): footprint cutoff = 3.184 cycles/m, sampling
   Nyquist (video-frame-matched) = 2.397 cycles/m — sampling Nyquist is lower,
   used as the conservative gate per the brief's own rule. Still no valid band
   above it (freqs max out exactly at that same Nyquist by construction).
7. **Checked the altimeter's own true native ping rate directly from the raw
   file timestamps** (not our video-frame extraction choice): 19,048 readings,
   median inter-ping interval 0.2553s → **3.92 Hz**, matching its datasheet's
   documented "4Hz" data-rate option exactly. At this survey's ~1 m/s ROV
   speed, that's a native along-track spacing of ~0.265m — the altimeter's
   *own* sampling Nyquist (~1.89 cycles/m) is *below* its *own* footprint
   cutoff (3.18 cycles/m). **This is not fixable by denser video extraction or
   better rendering** — the sonar itself, as flown at this range, does not
   sample its own beam footprint at Nyquist. There is no real frequency band
   in which "does vision exceed sonar above cutoff" is even a valid question
   to ask of this recording.

## Why this happens specifically here, and why it might not elsewhere

The footprint cutoff scales inversely with range (`d = 2 r tan(3°)`), so it
gets *tighter* — harder to sample adequately — the *closer* the survey flies.
PS117-39 (mean range 1.50m) and PS117-29 (13cm close approach) are both very
close-range surveys by construction (these are the two tiers already used
throughout this project). At the brief's own stated "standard" survey altitude
of ~20m, the footprint would be ~2.09m (`k_cut≈0.24 cycles/m`) — comfortably
coarse enough for a 3.92Hz altimeter at ROV survey speed to resolve above-cutoff
structure easily. **The real-data spectral test is range-dependent, and this
project's two existing real-data tiers happen to both be in the range regime
where a modest single-beam altimeter structurally cannot support it.**

## Recommendation

Claim C's real-data spectral proof cannot be completed with the data in hand,
regardless of further engineering effort on this project's side — this is a
sensor-recording limitation, not a pipeline limitation. Two live paths:
1. **Task 5 (simulator)** licenses claim C properly: a synthetic DT101 (or
   even a synthetic denser altimeter) can be sampled at whatever rate the test
   needs, at controlled range, with a true GT mesh spectrum to validate against
   — this becomes the primary evidence for claim C, not a secondary check.
2. If real-data validation is still wanted, it needs a *farther-range* real
   sequence than either existing tier (closer to that ~20m regime) where this
   same altimeter's real 3.92Hz rate would actually resolve the footprint band
   — worth flagging as a Task 4 sequence-selection criterion if any candidate
   windows at longer range exist in the raw PS117-29/37/39 archives.

## Split-half coherence control

Still valid and running independently of this verdict (doesn't depend on
k_cut) — two 30k reconstructions training now on alternating-frame halves,
config identical to the 30k headline model. Status: in progress.
