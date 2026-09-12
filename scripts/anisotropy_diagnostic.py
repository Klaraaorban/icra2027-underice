"""Gaussian anisotropy as a sensor-free geometric-error diagnostic. Reads a
trained model's point_cloud.ply directly (no GPU/CUDA needed -- pure PLY
parsing), computes per-Gaussian scale anisotropy (ratio of largest to
smallest of the 3 scale axes, after the exp() activation the model actually
uses), and the "needle fraction" (fraction of Gaussians above a ratio
threshold). Correlates needle fraction against sonar r across configs -- if
it predicts geometric error, this is a diagnostic usable with no sonar at
all, since it only reads the model's own saved parameters.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.stats import spearmanr


def needle_fraction(ply_path, ratio_threshold=10.0):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    scales = np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]), np.asarray(v["scale_2"])], axis=1)
    scales = np.exp(scales)  # raw stored values are log-scale
    ratios = scales.max(axis=1) / np.clip(scales.min(axis=1), 1e-9, None)
    frac = float((ratios > ratio_threshold).mean())
    return frac, float(np.median(ratios)), int(len(ratios))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-json", required=True,
                         help="JSON: list of {label, ply_path, sonar_r}")
    parser.add_argument("--ratio-threshold", type=float, default=10.0)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    with open(args.configs_json) as f:
        configs = json.load(f)

    rows = []
    for c in configs:
        ply_path = Path(c["ply_path"])
        if not ply_path.exists():
            print(f"MISSING: {c['label']} -> {ply_path}")
            continue
        frac, median_ratio, n = needle_fraction(ply_path, args.ratio_threshold)
        rows.append({"label": c["label"], "sonar_r": c["sonar_r"], "needle_fraction": frac,
                     "median_ratio": median_ratio, "n_gaussians": n})
        print(f"{c['label']:35s} n={n:7d}  needle_frac(>{args.ratio_threshold:.0f}x)={frac:.4f}  "
              f"median_ratio={median_ratio:.2f}  sonar_r={c['sonar_r']}")

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "sonar_r", "needle_fraction", "median_ratio", "n_gaussians"])
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["sonar_r"] is not None]
    if len(valid) >= 3:
        needle = np.array([r["needle_fraction"] for r in valid])
        sonar = np.array([r["sonar_r"] for r in valid])
        rho, p = spearmanr(needle, sonar)
        print(f"\nn={len(valid)}  Spearman rho(needle_fraction, sonar_r) = {rho:.3f}  (p={p:.4f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
