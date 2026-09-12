#!/bin/bash
# Run after run_f1_holdout_training.sh finishes.
# Syncs F1 results to splatt_enhancement repo and commits.
set -e

RELEASE="E:/Research/Holo/icra2027_underice/release/sonar_eval"
SPLATT="/c/Users/orban/AppData/Local/Temp/claude/splatt_enhancement/sonar_eval"
F1_JSON="${RELEASE}/results_ours_f1.json"

if [ ! -f "$F1_JSON" ]; then
  echo "ERROR: $F1_JSON not found -- training not finished yet?"
  exit 1
fi

echo "F1 results found. Syncing..."
cp "$F1_JSON" "$SPLATT/"

# Show summary
python3 -c "
import json
d = json.load(open('${F1_JSON}'))
print('F1 5-fold CV (truly held-out sonar training):')
print(f'  r = {d[\"test_r_mean\"]:.3f} +/- {d[\"test_r_std\"]:.3f}')
print(f'  RMSE = {d[\"test_rmse_mean\"]:.3f} +/- {d[\"test_rmse_std\"]:.3f} m')
for k, (r, rmse) in enumerate(zip(d[\"per_block_r\"], d[\"per_block_rmse\"])):
    print(f'  block {k}: r={r:.3f}, rmse={rmse:.3f} m')
"

echo ""
echo "Next steps:"
echo "  1. Update release/README.md with F1 numbers"
echo "  2. Regenerate Sonar_Story.pdf"
echo "  3. cd /c/Users/orban/AppData/Local/Temp/claude/splatt_enhancement && git add sonar_eval/results_ours_f1.json && git commit -m 'F1: truly block-held-out sonar CV results' && git push"
