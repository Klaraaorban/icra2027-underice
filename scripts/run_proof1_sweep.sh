#!/bin/bash
# Proof 1 sweep orchestration: seed 0 across all 5 conditions, both stations,
# before starting seed 1 (per the task's explicit instruction, so a partial
# run still yields a complete curve). 7000 iterations per run (disclosed
# deliberate choice -- see session notes -- 30 runs at 30k would be ~22+ hours).
# deliberately NOT using `set -e` -- each run's failure is handled explicitly
# below so one bad run doesn't kill the rest of a 30-run sweep
GS="E:/Research/Holo/gaussian-splatting"
ICRA="E:/Research/Holo/icra2027_underice"
PY="/c/Users/orban/miniconda3/envs/gaussian_splatting/python.exe"
cd "$GS"

run_one () {
  station=$1; seed=$2; cond=$3; source_path=$4; sonar_scale=$5; extra_depth_args=$6
  name="${station}_seed${seed}_${cond}"
  outdir="output/proof1/${name}"
  sonar_json="${source_path}/proof1_decimated/seed${seed}_${cond}_sonar_depths.json"
  test_names="${source_path}/proof1_decimated/seed${seed}_test_names.json"
  eval_json="${ICRA}/results/runs/proof1/${name}.json"
  mkdir -p "$(dirname "$eval_json")"

  if [ -f "$eval_json" ]; then
    echo "[$(date)] SKIP $name (already evaluated)"
    return
  fi

  echo "[$(date)] TRAIN $name"
  if [ "$cond" == "none" ]; then
    sonar_args="--sonar_loss_weight 0"
  else
    sonar_args="--sonar_depths_json $sonar_json --sonar_scale $sonar_scale --sonar_loss_weight 0.1 --sonar_adaptive_gain 3.0"
  fi
  "$PY" train.py -s "$source_path" -m "$outdir" --eval --data_device cpu \
    $extra_depth_args $sonar_args \
    --iterations 7000 --densify_until_iter 5000 \
    --test_iterations 7000 --save_iterations 7000 \
    > "output/proof1/${name}_run.log" 2>&1
  train_exit=$?
  if [ $train_exit -ne 0 ]; then
    echo "[$(date)] TRAIN FAILED $name (exit $train_exit) -- see output/proof1/${name}_run.log"
    return
  fi

  echo "[$(date)] EVAL $name"
  # deliberately NOT borrowing pings from a different condition to fit "none"'s
  # scale -- a true zero-ping deployment has no way to establish metric scale
  # at all, so test_r/rmse should come back null for "none", not a number
  # calibrated on pings that condition never actually had access to (rule 2)
  "$PY" "${ICRA}/scripts/proof1_evaluate.py" \
    --config "${ICRA}/configs/${station}.yaml" \
    --model-path "$outdir" --iteration 7000 \
    --test-names-json "$test_names" \
    --train-sonar-depths-json "$sonar_json" \
    --seed "$seed" --condition "$cond" --station "$station" \
    --out-json "$eval_json" \
    > "output/proof1/${name}_eval.log" 2>&1 || echo "[$(date)] EVAL FAILED $name -- see output/proof1/${name}_eval.log"
}

for seed in 0 1 2; do
  for cond in full every2nd every5th every10th none; do
    run_one ps117_39 $seed $cond \
      "data/ps117_39_30to90_fisheye" 5.2421 \
      "--depths depths_da3 --texture_confidence texture_confidence_calibrated"
  done
  for cond in full every2nd every5th every10th none; do
    run_one ps117_29 $seed $cond \
      "data/ps117_29_900to960_verified" 3.7511 \
      "--depths depths_da3 --texture_confidence texture_confidence_calibrated"
  done
  echo "[$(date)] === seed $seed complete across both stations ==="
done
echo "[$(date)] === PROOF 1 SWEEP COMPLETE ==="
