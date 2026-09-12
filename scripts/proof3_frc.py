"""Proof 3: resolution-without-ground-truth via Fourier Ring Correlation (1D
analogue), applied to the two independent split-half reconstructions already
trained for Task 0's coherence control (even-frame-only vs odd-frame-only,
same config, both renderable at all 300 original poses).

This is the established cryo-EM / super-resolution-microscopy method for
estimating a reconstruction's achieved resolution without independent ground
truth: two INDEPENDENT reconstructions of the same object should agree
(FRC near 1) up to the true achievable resolution, and disagree (FRC -> 0)
beyond it, since each one's noise/error beyond that point is independent of
the other's. Standard references: Saxton & Baumeister 1982 (introducing FRC
for EM); van Heel & Schatz 2005 (the 0.143 threshold as the now-standard
cryo-EM resolution criterion, chosen to correspond to a fixed SNR); Nieuwenhuizen
et al. 2013 ("Measuring image resolution in optical nanoscopy", Nat. Methods --
the extension of this same FRC methodology to optical/fluorescence microscopy,
the most direct methodological precedent for applying it to an optical 3D
reconstruction here). Classical FRC is defined over 2D Fourier rings (hence the
name); this is the 1D analogue along a single track, so each "ring" degenerates
to a single frequency bin -- correspondingly noisier per-bin, so a small local
moving-average smoothing is applied before reading off the resolution crossing,
disclosed explicitly rather than hidden.

Must run in the gaussian_splatting conda env.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_test_1d import render_all_frames_with_pose, fit_plane, detrend_poly  # noqa: E402
from split_half_coherence import extract_along_track_height  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def frc_1d(sig_a, sig_b, spacing, nperseg=64, noverlap=None):
    """Welch-segmented 1D FRC: FRC(k) = <Re(Fa Fb*)> / sqrt(<|Fa|^2><|Fb|^2>),
    averaged over overlapping windows -- NOT a single full-length transform
    with post-hoc bin smoothing. A single-transform FRC was tried first and
    rejected: in low-signal-power bands its raw values swing between +1 and -1
    from bin to bin (numerator dominated by phase noise once power is low),
    which a local moving-average over adjacent bins does NOT fix, since
    adjacent bins share the same noise-dominated regime rather than being
    independent samples of it. Proper segment averaging (used by
    scipy.signal.coherence, and mathematically FRC ~= sqrt(coherence) under
    matched averaging) gives each frequency estimate genuine independent
    samples to average over, which is what actually suppresses that noise.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    n = len(sig_a)
    step = nperseg - noverlap
    window = np.hanning(nperseg)

    cross_sum = None
    pa_sum = None
    pb_sum = None
    n_segs = 0
    for start in range(0, n - nperseg + 1, step):
        seg_a = sig_a[start:start + nperseg] * window
        seg_b = sig_b[start:start + nperseg] * window
        fa = np.fft.rfft(seg_a)
        fb = np.fft.rfft(seg_b)
        cross = fa * np.conj(fb)
        pa = np.abs(fa) ** 2
        pb = np.abs(fb) ** 2
        if cross_sum is None:
            cross_sum, pa_sum, pb_sum = cross.copy(), pa.copy(), pb.copy()
        else:
            cross_sum += cross
            pa_sum += pa
            pb_sum += pb
        n_segs += 1

    freqs = np.fft.rfftfreq(nperseg, d=spacing)
    denom = np.sqrt(pa_sum * pb_sum)
    frc = np.divide(np.real(cross_sum), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    print(f"FRC computed from {n_segs} overlapping segments of {nperseg} points each "
          f"(step={step}, {n} total points)")
    return freqs, frc, frc  # raw==smooth now; kept as 3-tuple for caller compatibility


def resolution_at_threshold(freqs, frc_smooth, threshold):
    below = frc_smooth < threshold
    for i in range(1, len(freqs)):
        if below[i] and np.all(below[i:min(i + 5, len(freqs))]):  # sustained drop, not a single noisy dip
            return float(freqs[i])
    return None  # never drops below threshold (or only transiently) -- resolution exceeds Nyquist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--mean-range-m", type=float, required=True,
                         help="From configs/sensor.yaml's per_sequence_mean_range")
    parser.add_argument("--beam-half-width-deg", type=float, default=None,
                         help="Only if independently sourced -- see TASK0_VERDICT.md; "
                              "omit to report footprint comparison as UNKNOWN")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ref_rows, sonar_scale = render_all_frames_with_pose(
        args.reference_model, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"),
    )
    cam_centers = np.array([r["cam_center"] for r in ref_rows])
    sonar_m = np.array([r["sonar_m"] for r in ref_rows])
    forwards = np.array([r["forward"] for r in ref_rows])
    ice_pts_sonar = cam_centers + (sonar_m / sonar_scale)[:, None] * forwards
    centroid, normal = fit_plane(ice_pts_sonar)
    cam_rel = cam_centers - centroid
    cam_in_plane = cam_rel - np.outer(cam_rel @ normal, normal)
    _, _, vt = np.linalg.svd(cam_in_plane)
    track_dir = vt[0]

    s_a, h_a = extract_along_track_height(args.model_a, args.iteration, cfg, centroid, normal, track_dir, sonar_scale)
    s_b, h_b = extract_along_track_height(args.model_b, args.iteration, cfg, centroid, normal, track_dir, sonar_scale)

    s_lo, s_hi = max(s_a.min(), s_b.min()), min(s_a.max(), s_b.max())
    spacing = min(np.median(np.diff(s_a)), np.median(np.diff(s_b)))
    s_uniform = np.arange(s_lo, s_hi, spacing)
    h_a_u = detrend_poly(s_uniform, np.interp(s_uniform, s_a, h_a), degree=2)
    h_b_u = detrend_poly(s_uniform, np.interp(s_uniform, s_b, h_b), degree=2)

    freqs, frc_raw, frc_smooth = frc_1d(h_a_u, h_b_u, spacing)

    res_0143 = resolution_at_threshold(freqs, frc_smooth, 0.143)
    res_05 = resolution_at_threshold(freqs, frc_smooth, 0.5)

    footprint_d = None
    if args.beam_half_width_deg is not None:
        footprint_d = 2 * args.mean_range_m * np.tan(np.radians(args.beam_half_width_deg))

    result = {
        "model_a": args.model_a, "model_b": args.model_b,
        "n_points": int(len(s_uniform)), "spacing_m": float(spacing),
        "freqs_cycles_per_m": freqs.tolist(), "frc_raw": frc_raw.tolist(), "frc_smooth": frc_smooth.tolist(),
        "resolution_at_0.143_criterion_cyc_per_m": res_0143,
        "resolution_at_0.143_criterion_cm": (100.0 / (2 * res_0143)) if res_0143 else None,
        "resolution_at_0.5_criterion_cyc_per_m": res_05,
        "resolution_at_0.5_criterion_cm": (100.0 / (2 * res_05)) if res_05 else None,
        "mean_range_m": args.mean_range_m,
        "beam_footprint_diameter_m": footprint_d,
        "beam_footprint_status": "computed" if footprint_d is not None else "UNKNOWN (no independently-sourced beam width provided)",
        "method_lineage": "Saxton & Baumeister 1982; van Heel & Schatz 2005 (0.143 criterion); "
                           "Nieuwenhuizen et al. 2013 Nat. Methods (FRC for optical nanoscopy -- "
                           "direct methodological precedent for this application)",
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"resolution @ 0.143 criterion: {res_0143} cyc/m "
          f"({result['resolution_at_0.143_criterion_cm']} cm)" if res_0143 else "0.143 criterion never sustained-crossed")
    print(f"resolution @ 0.5 criterion:   {res_05} cyc/m "
          f"({result['resolution_at_0.5_criterion_cm']} cm)" if res_05 else "0.5 criterion never sustained-crossed")
    if footprint_d is not None:
        print(f"altimeter footprint diameter at {args.mean_range_m}m: {footprint_d*100:.1f} cm")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
