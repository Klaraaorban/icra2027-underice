"""The headline decoupling statistic: Spearman rank correlation between PSNR
and held-out sonar r, computed ACROSS every distinct pipeline config tried on
each scene. Turns "PSNR doesn't track geometry" from an inference a reader has
to make by comparing two tables into one number per scene.

Reads results/runs/psnr_vs_sonar_configs.json (every (psnr, sonar_r) pair this
project has produced, one entry per distinct recipe -- see that file's
"provenance" field for exactly how it was assembled and what's collapsed to
avoid pseudoreplication).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ICRA = Path("E:/Research/Holo/icra2027_underice")


def main():
    with open(ICRA / "results/runs/psnr_vs_sonar_configs.json") as f:
        data = json.load(f)

    results = {}
    for station in ["ps117_39", "ps117_29"]:
        rows = data[station]
        psnr = np.array([r["psnr"] for r in rows])
        sonar_r = np.array([r["sonar_r"] for r in rows])
        rho, p = spearmanr(psnr, sonar_r)
        results[station] = {
            "n_configs": len(rows), "spearman_rho": float(rho), "p_value": float(p),
            "configs": [r["config"] for r in rows],
        }
        print(f"{station}: n={len(rows)}  Spearman rho(PSNR, sonar_r) = {rho:.3f}  (p={p:.4f})")

    out_path = ICRA / "results/runs/psnr_vs_sonar_rank_correlation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
