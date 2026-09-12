"""Backscatter-as-geometry check: are there low-opacity Gaussians floating
near the camera trajectory, consistent with 3DGS representing the water
column itself as geometry (rather than just the ice)? Pure PLY + camera-pose
read, no GPU needed for the descriptive statistic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-path", required=True)
    parser.add_argument("--cam-centers-npy", required=True,
                         help="Nx3 array of camera centres in the SAME coordinate frame as the ply (COLMAP units)")
    parser.add_argument("--near-dist", type=float, default=0.3, help="COLMAP-unit distance threshold to count as 'near the camera path'")
    parser.add_argument("--low-opacity-thresh", type=float, default=0.3)
    args = parser.parse_args()

    ply = PlyData.read(args.ply_path)
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacity_raw = np.asarray(v["opacity"])
    opacity = sigmoid(opacity_raw)  # 3DGS stores opacity pre-sigmoid

    cam_centers = np.load(args.cam_centers_npy)
    # nearest-camera distance per Gaussian, chunked to avoid a huge NxM matrix at once
    n = xyz.shape[0]
    min_dist = np.full(n, np.inf)
    chunk = 20000
    for i in range(0, n, chunk):
        d = np.linalg.norm(xyz[i:i+chunk, None, :] - cam_centers[None, :, :], axis=2)
        min_dist[i:i+chunk] = d.min(axis=1)

    near_camera = min_dist < args.near_dist
    low_opacity = opacity < args.low_opacity_thresh

    frac_near = float(near_camera.mean())
    frac_low_opacity_overall = float(low_opacity.mean())
    frac_low_opacity_near = float(low_opacity[near_camera].mean()) if near_camera.sum() > 0 else None
    frac_low_opacity_far = float(low_opacity[~near_camera].mean()) if (~near_camera).sum() > 0 else None

    print(f"n Gaussians: {n}")
    print(f"fraction within {args.near_dist} units of a camera centre: {frac_near:.4f}")
    print(f"fraction low-opacity (<{args.low_opacity_thresh}) overall: {frac_low_opacity_overall:.4f}")
    print(f"fraction low-opacity among NEAR-camera Gaussians: {frac_low_opacity_near}")
    print(f"fraction low-opacity among FAR-from-camera Gaussians: {frac_low_opacity_far}")
    if frac_low_opacity_near is not None and frac_low_opacity_far is not None:
        ratio = frac_low_opacity_near / max(frac_low_opacity_far, 1e-9)
        print(f"ratio (near-camera low-opacity rate / far low-opacity rate): {ratio:.2f}x")


if __name__ == "__main__":
    main()
