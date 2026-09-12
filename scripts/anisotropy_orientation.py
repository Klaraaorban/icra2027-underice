"""Anisotropy diagnostic, corrected with an orientation term. Distinguishes
the two things a raw scale-ratio can't: a surface-aligned disc (minor axis
~parallel to the local viewing ray -- fine, expected for a well-converged
flat surface) from a depth streak (major axis ~parallel to the viewing ray --
the actual "billboard" pathology). Pure PLY + camera-pose read, no GPU.

Quaternion convention verified directly from utils/general_utils.py's
build_rotation: stored as (w,x,y,z) in scene/gaussian_model.py's
rot_0..rot_3 properties; R's columns are where each LOCAL axis points in
world space (R @ local = world), so column[minor_scale_idx] IS the minor
axis's world-space direction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.stats import spearmanr


def quat_to_rotmat_batch(q):
    """q: (N,4) as (w,x,y,z), normalized. Returns (N,3,3), columns = local axis directions in world space."""
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(norm, 1e-9, None)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def analyze_one(ply_path, cam_centers, ratio_threshold, angle_threshold_deg):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    scales = np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]), np.asarray(v["scale_2"])], axis=1)
    scales = np.exp(scales)
    quat = np.stack([np.asarray(v["rot_0"]), np.asarray(v["rot_1"]), np.asarray(v["rot_2"]), np.asarray(v["rot_3"])], axis=1)

    ratios = scales.max(axis=1) / np.clip(scales.min(axis=1), 1e-9, None)
    anisotropic = ratios > ratio_threshold
    n_aniso = int(anisotropic.sum())
    if n_aniso == 0:
        return {"n_gaussians": len(xyz), "n_anisotropic": 0}

    idx = np.where(anisotropic)[0]
    xyz_a = xyz[idx]
    scales_a = scales[idx]
    quat_a = quat[idx]

    R = quat_to_rotmat_batch(quat_a)  # (n_aniso, 3, 3)
    minor_axis_idx = scales_a.argmin(axis=1)
    major_axis_idx = scales_a.argmax(axis=1)
    minor_dir = R[np.arange(n_aniso), :, minor_axis_idx]  # (n_aniso, 3)
    major_dir = R[np.arange(n_aniso), :, major_axis_idx]

    # nearest camera's viewing ray at each anisotropic Gaussian's position
    # chunked nearest-neighbour to avoid a huge distance matrix at once
    nearest_idx = np.zeros(n_aniso, dtype=np.int64)
    chunk = 5000
    for i in range(0, n_aniso, chunk):
        d = np.linalg.norm(xyz_a[i:i+chunk, None, :] - cam_centers[None, :, :], axis=2)
        nearest_idx[i:i+chunk] = d.argmin(axis=1)
    ray = xyz_a - cam_centers[nearest_idx]
    ray = ray / np.clip(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9, None)

    def angle_to_ray(axis_dir):
        cos_a = np.abs(np.sum(axis_dir * ray, axis=1))  # abs: axis direction is sign-ambiguous
        cos_a = np.clip(cos_a, -1, 1)
        return np.degrees(np.arccos(cos_a))

    minor_angle = angle_to_ray(minor_dir)   # ~0 deg = minor axis parallel to ray = disc, good
    major_angle = angle_to_ray(major_dir)   # ~0 deg = major axis parallel to ray = streak, bad

    disc_like = minor_angle < angle_threshold_deg
    streak_like = major_angle < angle_threshold_deg

    return {
        "n_gaussians": len(xyz), "n_anisotropic": n_aniso,
        "frac_anisotropic": n_aniso / len(xyz),
        "frac_disc_like_of_anisotropic": float(disc_like.mean()),
        "frac_streak_like_of_anisotropic": float(streak_like.mean()),
        "frac_streak_like_of_all": float(streak_like.sum() / len(xyz)),
        "mean_minor_angle_deg": float(minor_angle.mean()),
        "mean_major_angle_deg": float(major_angle.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-json", required=True, help="JSON: list of {label, ply_path, cam_centers_npy, sonar_r}")
    parser.add_argument("--ratio-threshold", type=float, default=10.0)
    parser.add_argument("--angle-threshold-deg", type=float, default=30.0)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    with open(args.configs_json) as f:
        configs = json.load(f)

    rows = []
    for c in configs:
        ply_path = Path(c["ply_path"])
        if not ply_path.exists():
            print(f"MISSING: {c['label']}")
            continue
        cam_centers = np.load(c["cam_centers_npy"])
        result = analyze_one(ply_path, cam_centers, args.ratio_threshold, args.angle_threshold_deg)
        result["label"] = c["label"]
        result["sonar_r"] = c["sonar_r"]
        rows.append(result)
        print(f"{c['label']:35s} n_aniso={result.get('n_anisotropic', 0):7d}  "
              f"streak_frac={result.get('frac_streak_like_of_anisotropic', 0):.4f}  "
              f"disc_frac={result.get('frac_disc_like_of_anisotropic', 0):.4f}  "
              f"sonar_r={c['sonar_r']}")

    import csv
    fieldnames = ["label", "sonar_r", "n_gaussians", "n_anisotropic", "frac_anisotropic",
                  "frac_disc_like_of_anisotropic", "frac_streak_like_of_anisotropic",
                  "frac_streak_like_of_all", "mean_minor_angle_deg", "mean_major_angle_deg"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    valid = [r for r in rows if r.get("sonar_r") is not None and "frac_streak_like_of_anisotropic" in r]
    if len(valid) >= 3:
        for metric in ["frac_streak_like_of_anisotropic", "frac_disc_like_of_anisotropic", "frac_streak_like_of_all"]:
            m = np.array([r[metric] for r in valid])
            s = np.array([r["sonar_r"] for r in valid])
            rho, p = spearmanr(m, s)
            print(f"\nn={len(valid)}  Spearman rho({metric}, sonar_r) = {rho:.3f}  (p={p:.4f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
