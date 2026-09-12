#!/bin/bash
GS="E:/Research/Holo/gaussian-splatting"
ICRA="E:/Research/Holo/icra2027_underice"
PY="/c/Users/orban/miniconda3/envs/gaussian_splatting/python.exe"
cd "$GS"

run_one () {
  name=$1; iteration=$2
  out="$ICRA/results/runs/psnr_extract/${name}.csv"
  if [ -f "$out" ]; then
    echo "[$(date)] SKIP $name"
    return
  fi
  echo "[$(date)] EXTRACT $name (iter $iteration)"
  "$PY" "$ICRA/scripts/extract_all_psnr.py" --model-path "output/$name" --iteration "$iteration" --out-csv "$out" \
    > "$ICRA/results/runs/psnr_extract/${name}.log" 2>&1 || echo "[$(date)] FAILED $name"
}

# PS117-39, standard images (ps117_39_30to90_fisheye)
run_one ps117_39_fisheye_eval 7000
run_one ps117_39_da3_depthreg 7000
run_one ps117_39_colmap_mast3r_fused 7000
run_one ps117_39_da3_texconf_gated 7000
run_one ps117_39_da3_texconf_sonar 7000
run_one ps117_39_da3_texconf_calibrated_sonar 7000
run_one ps117_39_da3_texconf_floor_sonar 7000
run_one ps117_39_selfdiag_sonar 7000

# PS117-39, MASt3R-seeded images (different image set)
run_one ps117_39_mast3r_seeded 7000
run_one ps117_39_mast3r_seeded_depthreg 7000

# PS117-29, standard COLMAP images (ps117_29_900to960_verified)
run_one ps117_29_texconf_gated 7000

# PS117-29, MASt3R-seeded images (different image set)
run_one ps117_29_mast3r_seeded_vanilla 7000
run_one ps117_29_mast3r_depthreg 7000
run_one ps117_29_mast3r_texconf_gated 7000
run_one ps117_29_mast3r_texconf_gated_30k 30000

echo "[$(date)] === ALL PSNR EXTRACTION COMPLETE ==="
