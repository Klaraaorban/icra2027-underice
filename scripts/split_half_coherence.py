"""Task 0's critical control: magnitude-squared coherence between two
independently-trained reconstructions (alternating-frame halves) along the
same track. Real structure should be reproducible (high coherence); noise
should not (coherence near zero). Doesn't depend on k_cut -- produces the full
coherence curve, which the eventual verdict (once the cutoff question settles,
see TASK0_VERDICT.md) gets compared against.

Both half-models were trained with the SAME config as the 30k headline model,
each supervised on only 150/300 alternating-parity frames, but all 300 camera
poses remain renderable in both (see train.py's --frame_parity addition).

Must run in the gaussian_splatting conda env.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import coherence as scipy_coherence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_test_1d import render_all_frames_with_pose, fit_plane, detrend_poly  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def extract_along_track_height(model_path, iteration, cfg, plane_centroid, plane_normal, track_dir, sonar_scale):
    rows, _ = render_all_frames_with_pose(
        model_path, iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"), sonar_scale=sonar_scale,
    )
    cam_centers = np.array([r["cam_center"] for r in rows])
    forwards = np.array([r["forward"] for r in rows])
    pred_units = np.array([r["pred_depth_units"] for r in rows])

    ice_pts = cam_centers + pred_units[:, None] * forwards
    cam_rel = cam_centers - plane_centroid
    cam_in_plane = cam_rel - np.outer(cam_rel @ plane_normal, plane_normal)
    s_units = cam_in_plane @ track_dir
    s_m = s_units * sonar_scale

    h = ((ice_pts - plane_centroid) @ plane_normal) * sonar_scale

    order = np.argsort(s_m)
    return s_m[order], h[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-a", required=True, help="e.g. output/ps117_39_coherence_even")
    parser.add_argument("--model-b", required=True, help="e.g. output/ps117_39_coherence_odd")
    parser.add_argument("--reference-model", required=True,
                         help="Full model used only to fix the shared ice-plane/track-axis/scale "
                              "so both halves are compared in the identical frame")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # fix a shared geometric frame from the reference (full-data) model, so
    # both halves are projected identically
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
    print(f"shared frame fixed from reference model: plane normal={normal}, sonar_scale={sonar_scale:.4f}")

    s_a, h_a = extract_along_track_height(args.model_a, args.iteration, cfg, centroid, normal, track_dir, sonar_scale)
    s_b, h_b = extract_along_track_height(args.model_b, args.iteration, cfg, centroid, normal, track_dir, sonar_scale)
    print(f"model A ({args.model_a}): {len(s_a)} points, s range [{s_a.min():.2f}, {s_a.max():.2f}] m")
    print(f"model B ({args.model_b}): {len(s_b)} points, s range [{s_b.min():.2f}, {s_b.max():.2f}] m")

    s_lo, s_hi = max(s_a.min(), s_b.min()), min(s_a.max(), s_b.max())
    spacing = min(np.median(np.diff(s_a)), np.median(np.diff(s_b)))
    s_uniform = np.arange(s_lo, s_hi, spacing)

    h_a_u = detrend_poly(s_uniform, np.interp(s_uniform, s_a, h_a), degree=2)
    h_b_u = detrend_poly(s_uniform, np.interp(s_uniform, s_b, h_b), degree=2)

    fs = 1.0 / spacing  # samples per metre -- "sampling frequency" in the spatial-frequency sense
    nperseg = min(64, len(s_uniform) // 4)
    freqs, coh = scipy_coherence(h_a_u, h_b_u, fs=fs, nperseg=nperseg)

    result = {
        "model_a": args.model_a, "model_b": args.model_b, "reference_model": args.reference_model,
        "n_points_shared_grid": int(len(s_uniform)), "spacing_m": float(spacing),
        "freqs_cycles_per_m": freqs.tolist(), "coherence": coh.tolist(),
        "mean_coherence_above_1cyc_per_m": float(np.mean(coh[freqs >= 1.0])) if np.any(freqs >= 1.0) else None,
        "mean_coherence_above_2cyc_per_m": float(np.mean(coh[freqs >= 2.0])) if np.any(freqs >= 2.0) else None,
        "pass_threshold": 0.5,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print("\nfreq (cyc/m)   coherence")
    for fq, c in zip(freqs, coh):
        flag = "  <-- >0.5" if c > 0.5 else ""
        print(f"  {fq:6.3f}       {c:.3f}{flag}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
