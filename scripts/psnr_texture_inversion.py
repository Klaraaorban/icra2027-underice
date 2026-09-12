"""Tests the hypothesis: PSNR tracks scene texture/entropy, not reconstruction
quality. Computes per-image Shannon entropy and gradient energy (mean Sobel
magnitude) for every ground-truth frame in both stations, then:
  1. Compares station-level mean texture metrics against station-level mean
     PSNR (does the low-texture station score the higher PSNR?).
  2. Within each station, correlates per-frame texture metrics against that
     frame's own PSNR (from the existing Proof-2 referee CSVs), a stronger
     test than the 2-point cross-scene comparison.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr


def entropy_and_gradient(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_energy = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    return entropy, grad_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir-39", required=True)
    parser.add_argument("--images-dir-29", required=True)
    parser.add_argument("--proof2-csv-39", required=True)
    parser.add_argument("--proof2-csv-29", required=True)
    parser.add_argument("--out-csv-39", required=True)
    parser.add_argument("--out-csv-29", required=True)
    args = parser.parse_args()

    station_means = {}
    for station, images_dir, proof2_csv, out_csv in [
        ("ps117_39", args.images_dir_39, args.proof2_csv_39, args.out_csv_39),
        ("ps117_29", args.images_dir_29, args.proof2_csv_29, args.out_csv_29),
    ]:
        psnr_by_name = {}
        with open(proof2_csv) as f:
            for row in csv.DictReader(f):
                psnr_by_name[row["name"]] = float(row["psnr_db"])

        rows = []
        for img_path in sorted(Path(images_dir).glob("*.jpg")):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            entropy, grad_energy = entropy_and_gradient(gray)
            name = img_path.stem
            psnr = psnr_by_name.get(name)
            rows.append({"name": name, "entropy": entropy, "grad_energy": grad_energy, "psnr": psnr})

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "entropy", "grad_energy", "psnr"])
            w.writeheader()
            w.writerows(rows)

        entropies = np.array([r["entropy"] for r in rows])
        grad_energies = np.array([r["grad_energy"] for r in rows])
        station_means[station] = {
            "mean_entropy": float(entropies.mean()), "mean_grad_energy": float(grad_energies.mean()),
            "mean_psnr_available": float(np.mean([r["psnr"] for r in rows if r["psnr"] is not None])),
            "n_images": len(rows),
        }
        print(f"=== {station} === n={len(rows)}  mean_entropy={entropies.mean():.3f}  "
              f"mean_grad_energy={grad_energies.mean():.3f}")

        matched = [r for r in rows if r["psnr"] is not None]
        if len(matched) >= 3:
            m_entropy = np.array([r["entropy"] for r in matched])
            m_grad = np.array([r["grad_energy"] for r in matched])
            m_psnr = np.array([r["psnr"] for r in matched])
            rho_e, p_e = spearmanr(m_entropy, m_psnr)
            rho_g, p_g = spearmanr(m_grad, m_psnr)
            print(f"  within-station (n={len(matched)}): "
                  f"rho(entropy, PSNR)={rho_e:.3f} (p={p_e:.4f})   "
                  f"rho(grad_energy, PSNR)={rho_g:.3f} (p={p_g:.4f})")
        print(f"  wrote {out_csv}")

    print("\n=== cross-scene comparison ===")
    for station, m in station_means.items():
        print(f"  {station}: mean_entropy={m['mean_entropy']:.3f}  mean_grad_energy={m['mean_grad_energy']:.3f}  "
              f"mean_psnr={m['mean_psnr_available']:.2f}")


if __name__ == "__main__":
    main()
