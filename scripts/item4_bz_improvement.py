"""Item 4, proper version: per-region B/Z correlated with per-region
geometric IMPROVEMENT from sonar supervision -- i.e. compares a sonar-
supervised model against its non-sonar twin (same recipe otherwise), per
window, using the delta in mean |residual|. Only meaningful on a scene where
sonar supervision was actually applied (PS117-39) -- PS117-29 never had it
enabled (the trust gate correctly disabled it), so there is no "improvement"
to measure there; that scene's simpler single-model result is reported
separately in item4_bz_per_region.py.

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
from item4_bz_per_region import perpendicular_baseline  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sonar-model-path", required=True)
    parser.add_argument("--nosonar-model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sonar-scale", type=float, required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows_sonar, _ = render_all_frames_with_pose(
        args.sonar_model_path, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"), sonar_scale=args.sonar_scale,
    )
    rows_nosonar, _ = render_all_frames_with_pose(
        args.nosonar_model_path, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"), sonar_scale=args.sonar_scale,
    )
    by_name_sonar = {r["orig_name"]: r for r in rows_sonar}
    by_name_nosonar = {r["orig_name"]: r for r in rows_nosonar}
    common_names = sorted(set(by_name_sonar) & set(by_name_nosonar), key=lambda n: by_name_sonar[n]["t_s"])
    print(f"n common frames: {len(common_names)}")

    windows = []
    for start in range(0, len(common_names) - args.window_size + 1, args.stride):
        names = common_names[start:start + args.window_size]
        chunk_sonar = [by_name_sonar[n] for n in names]
        chunk_nosonar = [by_name_nosonar[n] for n in names]

        cam_centers = np.array([r["cam_center"] for r in chunk_sonar])
        forwards = np.array([r["forward"] for r in chunk_sonar])
        sonar_vals = np.array([r["sonar_m"] for r in chunk_sonar])
        B = perpendicular_baseline(cam_centers, forwards) * args.sonar_scale
        Z = float(sonar_vals.mean())
        bz_ratio = B / Z if Z > 1e-9 else None

        pred_sonar_m = np.array([r["pred_depth_units"] for r in chunk_sonar]) * args.sonar_scale
        pred_nosonar_m = np.array([r["pred_depth_units"] for r in chunk_nosonar]) * args.sonar_scale
        resid_sonar = float(np.mean(np.abs(pred_sonar_m - sonar_vals)))
        resid_nosonar = float(np.mean(np.abs(pred_nosonar_m - sonar_vals)))
        improvement = resid_nosonar - resid_sonar  # positive = sonar supervision helped

        windows.append({
            "t_mid_s": float(np.mean([r["t_s"] for r in chunk_sonar])), "n_frames": len(names),
            "B_m": B, "Z_m": Z, "bz_ratio": bz_ratio,
            "resid_sonar_m": resid_sonar, "resid_nosonar_m": resid_nosonar,
            "improvement_m": improvement,
        })

    df = pd.DataFrame(windows)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string())

    valid = df.dropna(subset=["bz_ratio"])
    rho, p = spearmanr(valid["bz_ratio"], valid["improvement_m"])
    print(f"\nn windows: {len(valid)}")
    print(f"Spearman rho(B/Z, improvement from sonar supervision): {rho:.3f}  (p={p:.4f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
