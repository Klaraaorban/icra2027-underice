"""Expanded texture-inversion analysis: joins every available config's
per-frame PSNR (2 already-computed proof2 CSVs + 15 newly-extracted configs)
against the correct texture CSV for whichever image set that config actually
rendered against, then computes the correlation pooled and per-station,
across thousands of (frame, config) pairs instead of ~500.
"""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

RUNS = Path("E:/Research/Holo/icra2027_underice/results/runs")

# (psnr_csv, name_col, psnr_col, texture_csv, station) -- psnr_csv format varies:
# proof2 CSVs have header name,split,t_s,sonar_m,pred_m,residual_m,abs_residual_m,psnr_db,render_ms
# extract_all_psnr.py CSVs have header name,split,psnr
SOURCES = [
    ("ps117_39_30k_proof2_referee.csv", "psnr_db", "texture_ps117_39.csv", "ps117_39", "ps117_39_headline_30k"),
    ("ps117_29_vanilla_proof2_referee.csv", "psnr_db", "texture_ps117_29.csv", "ps117_29", "ps117_29_vanilla"),
    ("psnr_extract/ps117_39_fisheye_eval.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_fisheye_eval"),
    ("psnr_extract/ps117_39_da3_depthreg.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_da3_depthreg"),
    ("psnr_extract/ps117_39_colmap_mast3r_fused.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_colmap_mast3r_fused"),
    ("psnr_extract/ps117_39_da3_texconf_gated.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_da3_texconf_gated"),
    ("psnr_extract/ps117_39_da3_texconf_sonar.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_da3_texconf_sonar"),
    ("psnr_extract/ps117_39_da3_texconf_calibrated_sonar.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_da3_texconf_calibrated_sonar"),
    ("psnr_extract/ps117_39_da3_texconf_floor_sonar.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_da3_texconf_floor_sonar"),
    ("psnr_extract/ps117_39_selfdiag_sonar.csv", "psnr", "texture_ps117_39.csv", "ps117_39", "ps117_39_selfdiag_sonar"),
    ("psnr_extract/ps117_39_mast3r_seeded.csv", "psnr", "texture_ps117_39_mast3r.csv", "ps117_39", "ps117_39_mast3r_seeded"),
    ("psnr_extract/ps117_39_mast3r_seeded_depthreg.csv", "psnr", "texture_ps117_39_mast3r.csv", "ps117_39", "ps117_39_mast3r_seeded_depthreg"),
    ("psnr_extract/ps117_29_texconf_gated.csv", "psnr", "texture_ps117_29.csv", "ps117_29", "ps117_29_texconf_gated"),
    ("psnr_extract/ps117_29_mast3r_seeded_vanilla.csv", "psnr", "texture_ps117_29_mast3r.csv", "ps117_29", "ps117_29_mast3r_seeded_vanilla"),
    ("psnr_extract/ps117_29_mast3r_depthreg.csv", "psnr", "texture_ps117_29_mast3r.csv", "ps117_29", "ps117_29_mast3r_depthreg"),
    ("psnr_extract/ps117_29_mast3r_texconf_gated.csv", "psnr", "texture_ps117_29_mast3r.csv", "ps117_29", "ps117_29_mast3r_texconf_gated"),
    ("psnr_extract/ps117_29_mast3r_texconf_gated_30k.csv", "psnr", "texture_ps117_29_mast3r.csv", "ps117_29", "ps117_29_mast3r_texconf_gated_30k"),
]


def load_texture(path):
    d = {}
    with open(RUNS / path) as f:
        for row in csv.DictReader(f):
            d[row["name"]] = (float(row["entropy"]), float(row["grad_energy"]))
    return d


def main():
    texture_cache = {}
    pooled = {"ps117_39": [], "ps117_29": []}
    per_config = defaultdict(list)

    for psnr_csv, psnr_col, texture_csv, station, label in SOURCES:
        if texture_csv not in texture_cache:
            texture_cache[texture_csv] = load_texture(texture_csv)
        texture = texture_cache[texture_csv]

        n_matched = 0
        with open(RUNS / psnr_csv) as f:
            for row in csv.DictReader(f):
                name = row["name"]
                if name not in texture:
                    continue
                psnr = float(row[psnr_col])
                entropy, grad_energy = texture[name]
                pooled[station].append((entropy, grad_energy, psnr))
                per_config[label].append((entropy, grad_energy, psnr))
                n_matched += 1
        print(f"{label:40s} station={station:10s} n_matched={n_matched}")

    print("\n=== per-config correlations ===")
    for label, rows in per_config.items():
        if len(rows) < 3:
            continue
        e = np.array([r[0] for r in rows])
        g = np.array([r[1] for r in rows])
        p = np.array([r[2] for r in rows])
        rho_g, pval_g = spearmanr(g, p)
        print(f"  {label:40s} n={len(rows):4d}  rho(grad_energy,PSNR)={rho_g:+.3f} (p={pval_g:.4f})")

    print("\n=== POOLED per-station (all configs combined) ===")
    for station, rows in pooled.items():
        e = np.array([r[0] for r in rows])
        g = np.array([r[1] for r in rows])
        p = np.array([r[2] for r in rows])
        rho_e, pval_e = spearmanr(e, p)
        rho_g, pval_g = spearmanr(g, p)
        print(f"{station}: n={len(rows)}")
        print(f"  rho(entropy, PSNR)     = {rho_e:+.3f}  (p={pval_e:.2e})")
        print(f"  rho(grad_energy, PSNR) = {rho_g:+.3f}  (p={pval_g:.2e})")

    print("\n=== POOLED across BOTH stations ===")
    all_rows = pooled["ps117_39"] + pooled["ps117_29"]
    e = np.array([r[0] for r in all_rows])
    g = np.array([r[1] for r in all_rows])
    p = np.array([r[2] for r in all_rows])
    rho_e, pval_e = spearmanr(e, p)
    rho_g, pval_g = spearmanr(g, p)
    print(f"n={len(all_rows)}")
    print(f"  rho(entropy, PSNR)     = {rho_e:+.3f}  (p={pval_e:.2e})")
    print(f"  rho(grad_energy, PSNR) = {rho_g:+.3f}  (p={pval_g:.2e})")


if __name__ == "__main__":
    main()
