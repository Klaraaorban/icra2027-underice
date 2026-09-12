#!/bin/bash
# Sonar weight sensitivity sweep: same rigorous holdout as Proof 1 (seed0's
# contiguous test block, genuinely excluded from sonar supervision), same
# base recipe (DA3 depth-reg + calibrated confidence gate + adaptive gain
# 3.0) as the headline config, varying ONLY sonar_loss_weight.
GS="E:/Research/Holo/gaussian-splatting"
ICRA="E:/Research/Holo/icra2027_underice"
PY="/c/Users/orban/miniconda3/envs/gaussian_splatting/python.exe"
cd "$GS"

SONAR_JSON="data/ps117_39_30to90_fisheye/proof1_decimated/seed0_full_sonar_depths.json"
TEST_NAMES="data/ps117_39_30to90_fisheye/proof1_decimated/seed0_test_names.json"

run_one () {
  weight=$1
  name="ps117_39_weightsweep_w${weight}"
  outdir="output/weightsweep/${name}"
  eval_json="${ICRA}/results/runs/weightsweep/${name}.json"
  mkdir -p "$(dirname "$eval_json")"

  if [ -f "$eval_json" ]; then
    echo "[$(date)] SKIP $name"
    return
  fi

  echo "[$(date)] TRAIN $name (weight=$weight)"
  "$PY" train.py -s data/ps117_39_30to90_fisheye -m "$outdir" --eval --data_device cpu \
    --depths depths_da3 --texture_confidence texture_confidence_calibrated \
    --sonar_depths_json "$SONAR_JSON" --sonar_scale 5.2421 \
    --sonar_loss_weight "$weight" --sonar_adaptive_gain 3.0 \
    --iterations 7000 --densify_until_iter 5000 \
    --test_iterations 7000 --save_iterations 7000 \
    > "output/weightsweep/${name}_run.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date)] TRAIN FAILED $name -- see output/weightsweep/${name}_run.log"
    return
  fi

  echo "[$(date)] EVAL $name"
  "$PY" "${ICRA}/scripts/proof1_evaluate.py" \
    --config "${ICRA}/configs/ps117_39.yaml" \
    --model-path "$outdir" --iteration 7000 \
    --test-names-json "$TEST_NAMES" \
    --train-sonar-depths-json "$SONAR_JSON" \
    --seed 0 --condition "w${weight}" --station ps117_39 \
    --out-json "$eval_json" \
    > "output/weightsweep/${name}_eval.log" 2>&1 || echo "[$(date)] EVAL FAILED $name"
}

for weight in 0.01 0.03 0.1 0.3 1.0 3.0 10.0; do
  run_one $weight
done
echo "[$(date)] === WEIGHT SWEEP COMPLETE ==="
