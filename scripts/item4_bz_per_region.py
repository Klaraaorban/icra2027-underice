"""Item 4: per-region B/Z correlated with per-region geometric outcome, using
ONLY already-trained models (no new training). Gets well beyond n=2 by using
small sliding windows along the track instead of the 5 coarse Proof-1 blocks.

B (perpendicular baseline) and Z (mean range) definitions follow the original
project brief's Task 2: B is the spread of camera centres in a window,
projected onto the plane PERPENDICULAR to the window's mean viewing ray (not
raw max pairwise distance -- forward/axial ROV motion contributes almost
nothing to triangulation, only cross-track spread does). Z is the window's
mean measured sonar range (a real, measured quantity, not a model-dependent
one). Per-region geometric outcome is that window's mean |residual| between
rendered depth (train-fit scale) and sonar, using the ALREADY-COMPUTED
Proof-2 referee CSVs.

Must run in the gaussian_splatting conda env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_test_1d import render_all_frames_with_pose  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def perpendicular_baseline(cam_centers, forwards):
    mean_dir = forwards.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
    centroid = cam_centers.mean(axis=0)
    rel = cam_centers - centroid
    rel_perp = rel - np.outer(rel @ mean_dir, mean_dir)
    dists = np.linalg.norm(rel_perp, axis=1)
    return float(dists.max() - dists.min()) if len(dists) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sonar-scale", type=float, required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows, _ = render_all_frames_with_pose(
        args.model_path, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"), sonar_scale=args.sonar_scale,
    )
    rows.sort(key=lambda r: r["t_s"])
    n = len(rows)
    print(f"n frames: {n}")

    windows = []
    for start in range(0, n - args.window_size + 1, args.stride):
        chunk = rows[start:start + args.window_size]
        cam_centers = np.array([r["cam_center"] for r in chunk])
        forwards = np.array([r["forward"] for r in chunk])
        sonar = np.array([r["sonar_m"] for r in chunk])
        pred_units = np.array([r["pred_depth_units"] for r in chunk])
        pred_m = pred_units * args.sonar_scale

        B = perpendicular_baseline(cam_centers, forwards) * args.sonar_scale  # scale to metres
        Z = float(sonar.mean())
        bz_ratio = B / Z if Z > 1e-9 else None
        mean_abs_residual = float(np.mean(np.abs(pred_m - sonar)))
        t_mid = float(np.mean([r["t_s"] for r in chunk]))

        windows.append({
            "t_mid_s": t_mid, "n_frames": len(chunk), "B_m": B, "Z_m": Z,
            "bz_ratio": bz_ratio, "mean_abs_residual_m": mean_abs_residual,
        })

    df = pd.DataFrame(windows)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string())

    valid = df.dropna(subset=["bz_ratio"])
    rho, p = spearmanr(valid["bz_ratio"], valid["mean_abs_residual_m"])
    print(f"\nn windows: {len(valid)}")
    print(f"Spearman rho(B/Z, mean |residual|): {rho:.3f}  (p={p:.4f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
