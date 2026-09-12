#!/bin/bash
# Run after run_f1_holdout_training.sh + run_ablations.sh are both done.
# Syncs all new results to the splatt_enhancement repo and commits.
set -e

RELEASE="E:/Research/Holo/icra2027_underice/release"
SPLATT="/c/Users/orban/AppData/Local/Temp/claude/splatt_enhancement"
PY="/c/Users/orban/miniconda3/envs/gaussian_splatting/python.exe"
PY_GS="$PY"
GS="E:/Research/Holo/gaussian-splatting"
SONAR_EVAL="${RELEASE}/sonar_eval"
RESULTS="${SONAR_EVAL}"
SPLATT_SE="${SPLATT}/sonar_eval"
CONFIGS="E:/Research/Holo/icra2027_underice/configs"

echo "=== Checking required files ==="
MISSING=0
for f in \
  "${SONAR_EVAL}/results_ours_f1.json" \
  "${SONAR_EVAL}/results_ablation_nosonar.json" \
  "${SONAR_EVAL}/results_ablation_fixedweight.json" \
  "${SONAR_EVAL}/results_ablation_seasplat_30k.json" \
  "${SONAR_EVAL}/results_ablation.json"; do
  if [ -f "$f" ]; then
    echo "  OK: $(basename $f)"
  else
    echo "  MISSING: $f"
    MISSING=1
  fi
done
if [ "$MISSING" = "1" ]; then
  echo "Some files are missing. Run run_f1_holdout_training.sh and run_ablations.sh first."
  exit 1
fi

echo ""
echo "=== Syncing to splatt_enhancement ==="
cp "${SONAR_EVAL}/results_ours_f1.json" "$SPLATT_SE/"
cp "${SONAR_EVAL}/results_ablation_nosonar.json" "$SPLATT_SE/"
cp "${SONAR_EVAL}/results_ablation_fixedweight.json" "$SPLATT_SE/"
cp "${SONAR_EVAL}/results_ablation_seasplat_30k.json" "$SPLATT_SE/"
cp "${SONAR_EVAL}/results_ablation.json" "$SPLATT_SE/"
cp "${SONAR_EVAL}/README.md" "$SPLATT_SE/"
cp "${SONAR_EVAL}/eval_ours.py" "$SPLATT_SE/"
cp "${CONFIGS}/ps117_29.yaml" "$SPLATT_SE/configs/"
cp "${CONFIGS}/ps117_29_mast3r.yaml" "$SPLATT_SE/configs/"
cp "${RELEASE}/README.md" "$SPLATT/"
cp "${RELEASE}/METHODS.md" "$SPLATT/"
# PS117-29 results (may be absent if PS117-29 training hasn't run yet)
[ -f "${SONAR_EVAL}/results_ps29_baseline.json" ] && cp "${SONAR_EVAL}/results_ps29_baseline.json" "$SPLATT_SE/"
[ -f "${SONAR_EVAL}/results_ps29_ours.json" ]    && cp "${SONAR_EVAL}/results_ps29_ours.json" "$SPLATT_SE/"

echo ""
echo "=== F1 results ==="
"$PY" -c "
import json
d = json.load(open('${SONAR_EVAL}/results_ours_f1.json'))
print(f'  F1 r = {d[\"test_r_mean\"]:.3f} +/- {d[\"test_r_std\"]:.3f}')
print(f'  F1 RMSE = {d[\"test_rmse_mean\"]:.3f} +/- {d[\"test_rmse_std\"]:.3f} m')
"

echo ""
echo "=== Ablation table ==="
"$PY" -c "
import json
rows = json.load(open('${SONAR_EVAL}/results_ablation.json'))
print(f'  {\"Method\":<45} {\"r (mean +/- std)\":<22} RMSE')
print('  ' + '-'*75)
for r in rows:
    if r['test_r_mean'] is not None:
        print(f'  {r[\"label\"]:<45} {r[\"test_r_mean\"]:+.3f} +/- {r[\"test_r_std\"]:.3f}  {r[\"test_rmse_mean\"]:.3f} +/- {r[\"test_rmse_std\"]:.3f}')
    else:
        print(f'  {r[\"label\"]:<45} (missing)')
"

echo ""
echo "=== Committing to splatt_enhancement ==="
cd "$SPLATT"
git add sonar_eval/results_ours_f1.json \
        sonar_eval/results_ablation_nosonar.json \
        sonar_eval/results_ablation_fixedweight.json \
        sonar_eval/results_ablation_seasplat_30k.json \
        sonar_eval/results_ablation.json \
        sonar_eval/eval_ours.py \
        sonar_eval/README.md \
        sonar_eval/configs/ps117_29.yaml \
        sonar_eval/configs/ps117_29_mast3r.yaml \
        METHODS.md \
        README.md
# Optional PS117-29 results
git add sonar_eval/results_ps29_baseline.json 2>/dev/null || true
git add sonar_eval/results_ps29_ours.json 2>/dev/null || true
git commit -m "F1 truly-held-out + ablation sweep + PS117-29 multi-station results"
# git push  # skipped — run manually when ready to publish
echo ""
echo "=== Generating dense sonar trace ==="
cd "$GS"
if [ ! -f "${RESULTS}/dense_trace_ours.json" ]; then
  "$PY_GS" "${RESULTS}/gen_dense_trace.py" \
    --config "$CFG" --gs-root "$GS" \
    --model-path "$GS/output/ps117_39_da3_texconf_calibrated_sonar_30k" \
    --iteration 30000 --scale 5.2421 \
    --out "${RESULTS}/dense_trace_ours.json" \
    > "output/dense_trace_ours.log" 2>&1
  echo "[$(date)] Dense trace generated"
  cp "${RESULTS}/dense_trace_ours.json" "$SPLATT_SE/"
  git -C "$SPLATT" add sonar_eval/dense_trace_ours.json sonar_eval/gen_dense_trace.py 2>/dev/null || true
else
  echo "[$(date)] Dense trace already exists"
fi

echo ""
echo "Done. All results committed and pushed to splatt_enhancement."
