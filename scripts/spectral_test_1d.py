"""Task 0 (re-scoped): 1D along-track spectral go/no-go test for claim C.

Re-scoped per results/TASK0_TASK1_RESCOPE_DECISION.md: no multibeam swath exists
in this dataset, only a single-beam Valeport altimeter trace, so this compares
1D along-track profiles instead of 2D rasterized surfaces. Every quantity here
is either directly measured (camera poses from the trained model, altimeter
readings, rendered depth) or a standard, disclosed signal-processing step
(detrend, window, FFT) -- nothing is interpolated across gaps or fabricated.

Pipeline:
  1. Render every view's camera center + forward direction + center-patch depth,
     and match each frame to its altimeter reading (reuses acoustic_holdout's
     render_all_frame_depths, extended with pose data).
  2. Build a real 3D "ice surface point" per frame: camera_center + sonar_depth_m
     * forward_direction (both cameras are co-mounted and upward-looking per the
     project brief, so the altimeter's own boresight is assumed close enough to
     the camera's forward axis for this rigid co-mounted geometry -- disclosed
     assumption, not hidden).
  3. Fit a plane (least squares) to those 300 real 3D points -> ice-relative
     frame. Project camera centers onto that plane; take the dominant principal
     axis of the projected 2D positions as the along-track coordinate s.
  4. Two paired 1D signals vs s: h_sonar(s) (ice-point height above the fitted
     plane) and h_optical(s) (same, but using the RENDERED depth along the same
     boresight instead of the altimeter reading).
  5. Detrend each (2nd-degree polynomial vs s), apply a 1D Hann window, compute
     1D FFT power spectral density on a shared wavenumber axis (cycles/m).
  6. Cutoff: sampling-rate Nyquist (measured, from median along-track spacing)
     always computed; beam-footprint cutoff computed only if a beam width is
     supplied (BLOCKED for the real Valeport data -- see rescope doc).
  7. Report PASS/FAIL per the brief's criterion (excess energy factor >=3 over
     a decade of k above cutoff, not flat).

Must run in the gaussian_splatting conda env.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
from acoustic_holdout import render_all_frame_depths  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def render_all_frames_with_pose(model_path, iteration, valeport_path, clip_start_iso, fps,
                                 index_to_orig_name=None, sonar_scale=None):
    """Like acoustic_holdout.render_all_frame_depths, but also captures each
    view's world-space camera center and forward direction, needed for the
    plane fit / along-track axis. sonar_scale converts the altimeter's metres
    reading into the SAME COLMAP-unit depth scale the rendered depth is in, OR
    (if None) is fit here from all matched frames (not holdout-protected --
    this script tests spectral content, not held-out prediction accuracy, so
    using all available pairs for the one global scale constant is appropriate).
    """
    import bisect
    import torch
    from datetime import timedelta
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state

    def load_valeport(path):
        entries = []
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append((datetime.fromisoformat(d["timestamp"]), float(d["Ping Depth"])))
                except Exception:
                    pass
        entries.sort()
        return entries

    def nearest_ping(entries, timestamps, ts, max_gap=1.0):
        i = bisect.bisect_left(timestamps, ts)
        cands = [j for j in (i - 1, i) if 0 <= j < len(timestamps)]
        best = min(cands, key=lambda j: abs((timestamps[j] - ts).total_seconds()))
        gap = abs((timestamps[best] - ts).total_seconds())
        return entries[best][1] if gap <= max_gap else None

    name_remap = None
    if index_to_orig_name:
        with open(index_to_orig_name) as f:
            name_remap = json.load(f)

    _parser = _AP()
    model = ModelParams(_parser, sentinel=True)
    pipeline = PipelineParams(_parser)
    _parser.add_argument("--iteration", default=-1, type=int)
    sys.argv = ["x", "-m", str(model_path), "--iteration", str(iteration)]
    args = get_combined_args(_parser)
    safe_state(False)

    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    entries = load_valeport(valeport_path)
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(clip_start_iso)

    rows = []
    for views in (scene.getTrainCameras(), scene.getTestCameras()):
        for view in views:
            out = render(view, gaussians, pipeline.extract(args), bg)
            depth = out["depth"]
            _, h, w = depth.shape
            cy, cx = h // 2, w // 2
            center_depth_units = depth[0, cy - 10:cy + 10, cx - 10:cx + 10].mean().item()

            name = Path(view.image_name).stem
            orig_name = name_remap.get(name, name) if name_remap else name
            try:
                frame_idx = int(orig_name.lstrip("f"))
            except ValueError:
                continue
            t_s = frame_idx / fps
            est_ts = clip_start + timedelta(seconds=t_s)
            ping = nearest_ping(entries, timestamps, est_ts)
            if ping is None or ping == -99999:
                continue

            cam_center = view.camera_center.detach().cpu().numpy().astype(np.float64)
            cam_to_world = torch.inverse(view.world_view_transform).detach().cpu().numpy()
            forward = cam_to_world[:3, 2].astype(np.float64)
            forward = forward / (np.linalg.norm(forward) + 1e-12)

            rows.append({
                "name": name, "orig_name": orig_name, "t_s": t_s,
                "sonar_m": ping, "pred_depth_units": center_depth_units,
                "cam_center": cam_center.tolist(), "forward": forward.tolist(),
            })

    rows.sort(key=lambda r: r["t_s"])

    if sonar_scale is None:
        sonar = np.array([r["sonar_m"] for r in rows])
        pred = np.array([r["pred_depth_units"] for r in rows])
        sonar_scale = float(np.sum(sonar * pred) / max(np.sum(pred ** 2), 1e-9))

    return rows, sonar_scale


def fit_plane(points):
    """Least-squares plane through Nx3 points. Returns (centroid, normal)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
    if normal[2] < 0:  # keep "up" pointing roughly toward +z-ish world convention where possible
        normal = -normal
    return centroid, normal


def detrend_poly(s, y, degree=2):
    coeffs = np.polyfit(s, y, degree)
    trend = np.polyval(coeffs, s)
    return y - trend


def welch_like_psd_1d(s_uniform, y, spacing):
    """1D PSD via Hann-windowed FFT on a uniformly-resampled signal."""
    n = len(y)
    window = np.hanning(n)
    yw = y * window
    win_norm = np.sum(window ** 2)
    fft_vals = np.fft.rfft(yw)
    psd = (np.abs(fft_vals) ** 2) / (win_norm / spacing)
    freqs = np.fft.rfftfreq(n, d=spacing)  # cycles per metre
    return freqs, psd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--beam-half-width-deg", type=float, default=None,
                         help="Altimeter beam half-width, if known. Omit to leave the "
                              "footprint cutoff explicitly MISSING rather than guessed.")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows, sonar_scale = render_all_frames_with_pose(
        args.model_path, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"),
    )
    n = len(rows)
    print(f"n frames matched: {n}, fitted sonar_scale (COLMAP units -> m): {sonar_scale:.4f}")

    cam_centers = np.array([r["cam_center"] for r in rows])  # COLMAP units
    forwards = np.array([r["forward"] for r in rows])
    sonar_m = np.array([r["sonar_m"] for r in rows])
    pred_units = np.array([r["pred_depth_units"] for r in rows])
    pred_m = pred_units * sonar_scale

    # real 3D ice-surface points, in COLMAP units (cam_center in COLMAP units,
    # sonar_m converted to COLMAP units via the same scale for consistency)
    ice_pts_sonar = cam_centers + (sonar_m / sonar_scale)[:, None] * forwards
    centroid, normal = fit_plane(ice_pts_sonar)
    print(f"fitted ice-plane normal (COLMAP frame): {normal}")

    # along-track axis: dominant in-plane direction of camera centres
    cam_rel = cam_centers - centroid
    cam_in_plane = cam_rel - np.outer(cam_rel @ normal, normal)
    _, _, vt = np.linalg.svd(cam_in_plane)
    track_dir = vt[0]
    s_units = cam_in_plane @ track_dir  # along-track coordinate, COLMAP units
    s_m = s_units * sonar_scale  # convert to metres using the same fitted scale

    order = np.argsort(s_m)
    s_m = s_m[order]

    # heights above plane, in metres
    h_sonar = ((ice_pts_sonar - centroid) @ normal)[order] * sonar_scale
    ice_pts_optical = cam_centers + pred_units[:, None] * forwards
    h_optical = ((ice_pts_optical - centroid) @ normal)[order] * sonar_scale

    spacing_m = float(np.median(np.diff(s_m)))
    k_nyquist = 1.0 / (2 * spacing_m)
    print(f"median along-track spacing: {spacing_m:.4f} m, sampling Nyquist k: {k_nyquist:.3f} cycles/m")

    k_footprint = None
    if args.beam_half_width_deg is not None:
        mean_range = float(np.mean(sonar_m))
        footprint_d = 2 * mean_range * np.tan(np.radians(args.beam_half_width_deg))
        k_footprint = 1.0 / (2 * footprint_d)
        print(f"beam-footprint cutoff (mean range {mean_range:.2f}m): {k_footprint:.3f} cycles/m")
    else:
        print("beam-footprint cutoff: MISSING (no altimeter beam width available -- not guessed)")

    k_cut = k_nyquist if k_footprint is None else min(k_nyquist, k_footprint)
    print(f"conservative k_cut used for the gate: {k_cut:.3f} cycles/m")

    # resample onto a uniform along-track grid at the median spacing (standard,
    # required step for FFT of a not-perfectly-uniform real trajectory; NOT
    # interpolating across missing/invalid data -- every original point is real)
    s_uniform = np.arange(s_m.min(), s_m.max(), spacing_m)
    h_sonar_uniform = np.interp(s_uniform, s_m, h_sonar)
    h_optical_uniform = np.interp(s_uniform, s_m, h_optical)

    h_sonar_dt = detrend_poly(s_uniform, h_sonar_uniform, degree=2)
    h_optical_dt = detrend_poly(s_uniform, h_optical_uniform, degree=2)

    freqs_sonar, psd_sonar = welch_like_psd_1d(s_uniform, h_sonar_dt, spacing_m)
    freqs_optical, psd_optical = welch_like_psd_1d(s_uniform, h_optical_dt, spacing_m)

    above_cut = freqs_optical >= k_cut
    band_hi = min(freqs_optical.max(), k_cut * 10)  # "a decade of k above cutoff"
    band = above_cut & (freqs_optical <= band_hi)

    if band.sum() < 3:
        verdict = "INCONCLUSIVE"
        excess_factor = None
    else:
        ratio = psd_optical[band] / np.clip(psd_sonar[band], 1e-12, None)
        excess_factor = float(np.median(ratio))
        # "not flat white noise": check optical PSD actually decays/varies rather
        # than being flat across the band (std/mean of log-PSD as a flatness proxy)
        log_psd_band = np.log10(np.clip(psd_optical[band], 1e-12, None))
        flatness = float(np.std(log_psd_band))
        is_flat = flatness < 0.05
        verdict = "PASS" if (excess_factor >= 3.0 and not is_flat) else "FAIL"
        print(f"median excess factor P_optical/P_sonar above k_cut: {excess_factor:.2f}")
        print(f"optical PSD log10 std in band (flatness proxy, <0.05 ~ flat/noise-like): {flatness:.4f}")

    result = {
        "model_path": str(args.model_path),
        "iteration": args.iteration,
        "n_frames": n,
        "fitted_sonar_scale": sonar_scale,
        "along_track_spacing_m": spacing_m,
        "k_nyquist_sampling": k_nyquist,
        "k_footprint": k_footprint,
        "k_footprint_status": "computed" if k_footprint is not None else "MISSING_beam_width_not_documented",
        "k_cut_used": k_cut,
        "excess_factor_median": excess_factor,
        "verdict": verdict,
        "note_1d_not_2d": "This validates along-track spectral content only, not full 2D areal "
                           "coverage above cutoff -- see rescope decision doc.",
        "generated_at": datetime.now().isoformat(),
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    np.savez(str(Path(args.out_json).with_suffix(".npz")),
             s_uniform=s_uniform, h_sonar_dt=h_sonar_dt, h_optical_dt=h_optical_dt,
             freqs_sonar=freqs_sonar, psd_sonar=psd_sonar,
             freqs_optical=freqs_optical, psd_optical=psd_optical)

    print(f"\nVERDICT: {verdict}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
