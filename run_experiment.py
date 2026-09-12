"""
Experiment runner for ICRA 2027 sonar-supervised 3DGS experiments.

Usage:
    python run_experiment.py experiments/k_sweep.yaml
    python run_experiment.py experiments/adaptive_vs_fixed.yaml
    python run_experiment.py experiments/my_run.yaml --dry-run   # preview without training

Each YAML defines one experiment or a sweep. List values expand into a
cartesian product of runs. Results are appended to results/summary.csv.
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Station presets
# ---------------------------------------------------------------------------
STATIONS = {
    "ps117_39": {
        "source_path": "data/ps117_39_30to90_fisheye",
        "sonar_scale": 5.2421,
        "sonar_json_full": "data/ps117_39_30to90_fisheye/proof1_decimated/seed0_full_sonar_depths.json",
        "eval_config": "E:/Research/Holo/icra2027_underice/configs/ps117_39.yaml",
    },
    "ps117_29": {
        "source_path": "data/ps117_29_900to960_verified",
        "sonar_scale": 3.7511,
        "sonar_json_full": "data/ps117_29_900to960_verified/proof1_decimated/seed0_full_sonar_depths.json",
        "eval_config": "E:/Research/Holo/icra2027_underice/configs/ps117_29.yaml",
    },
}

# ---------------------------------------------------------------------------
# Paths (edit if your env differs)
# ---------------------------------------------------------------------------
GS_ROOT   = Path("E:/Research/Holo/gaussian-splatting")
PY        = Path("/c/Users/orban/miniconda3/envs/gaussian_splatting/python.exe")
EVAL_PY   = Path("E:/Research/Holo/icra2027_underice/release/sonar_eval/eval_ours.py")
RESULTS_DIR = Path("E:/Research/Holo/icra2027_underice/results")
CSV_PATH  = RESULTS_DIR / "summary.csv"

CSV_FIELDS = [
    "run_name", "station", "iterations", "sonar_loss_weight",
    "sonar_adaptive_gain", "dropout_pct", "pingrate_decimation",
    "test_r_mean", "test_r_std", "test_rmse_mean", "test_rmse_std",
    "train_r_mean", "n_frames", "started_at", "elapsed_s", "notes",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_sweep(spec: dict) -> list[dict]:
    """Expand any list-valued keys into a cartesian product of single-value dicts."""
    sweep_keys = [k for k, v in spec.items() if isinstance(v, list)]
    if not sweep_keys:
        return [spec]
    static = {k: v for k, v in spec.items() if k not in sweep_keys}
    combos = list(itertools.product(*[spec[k] for k in sweep_keys]))
    runs = []
    for combo in combos:
        run = dict(static)
        run.update(dict(zip(sweep_keys, combo)))
        runs.append(run)
    return runs


def run_name(spec: dict) -> str:
    base = spec.get("name", "exp")
    station = spec.get("station", "ps117_39")
    w = spec.get("sonar_loss_weight", 0.1)
    k = spec.get("sonar_adaptive_gain", 3.0)
    iters = spec.get("iterations", 30000)
    dropout = spec.get("dropout_pct", 0)
    decimation = spec.get("pingrate_decimation", 1)
    tag = f"w{w}_k{k}_i{iters}"
    if dropout:
        tag += f"_drop{dropout}"
    if decimation > 1:
        tag += f"_dec{decimation}"
    return f"{base}_{station}_{tag}"


def append_csv(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_subprocess(cmd: list, log_path: Path, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] logging to {log_path}")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(GS_ROOT), env={**os.environ, "PYTHONUTF8": "1"},
        )
        start = time.time()
        while proc.poll() is None:
            elapsed = int(time.time() - start)
            print(f"\r  [{label}] running ... {elapsed}s", end="", flush=True)
            time.sleep(15)
        print(f"\r  [{label}] done in {int(time.time()-start)}s        ")
        return proc.returncode


def select_sonar_json(spec: dict, station_cfg: dict) -> str:
    """Pick the right sonar JSON based on dropout_pct and pingrate_decimation."""
    dropout = spec.get("dropout_pct", 0)
    decimation = spec.get("pingrate_decimation", 1)

    # If an explicit path is given, use it
    if "sonar_json" in spec:
        return spec["sonar_json"]

    # Dropout: look for pre-generated dropout JSON (from dropout sweep)
    source = station_cfg["source_path"]
    if dropout > 0:
        cand = Path(GS_ROOT / source / "dropout_decimated" / f"dropout{dropout}_sonar_depths.json")
        if cand.exists():
            return str(cand)
        print(f"  WARNING: dropout JSON not found at {cand}, falling back to full")

    # Decimation: look for every-Nth ping JSON
    if decimation > 1:
        cand_name = f"seed0_every{decimation}th_sonar_depths.json"
        cand = Path(GS_ROOT / source / "proof1_decimated" / cand_name)
        if cand.exists():
            return str(cand)
        print(f"  WARNING: decimated JSON not found at {cand}, falling back to full")

    return str(GS_ROOT / station_cfg["sonar_json_full"])


# ---------------------------------------------------------------------------
# Core: train one model
# ---------------------------------------------------------------------------

def train(spec: dict, name: str, out_model_dir: Path, dry_run: bool) -> bool:
    station = spec.get("station", "ps117_39")
    station_cfg = STATIONS[station]
    sonar_json = select_sonar_json(spec, station_cfg)
    iters = spec.get("iterations", 30000)
    w = spec.get("sonar_loss_weight", 0.1)
    k = spec.get("sonar_adaptive_gain", 3.0)

    cmd = [
        str(PY), "train.py",
        "-s", station_cfg["source_path"],
        "-m", str(out_model_dir),
        "--eval", "--data_device", "cpu",
        "--depths", "depths_da3",
        "--texture_confidence", "texture_confidence_calibrated",
        "--sonar_depths_json", sonar_json,
        "--sonar_scale", str(station_cfg["sonar_scale"]),
        "--sonar_loss_weight", str(w),
        "--sonar_adaptive_gain", str(k),
        "--iterations", str(iters),
        "--densify_until_iter", str(min(15000, iters)),
        "--test_iterations", str(iters),
        "--save_iterations", str(iters),
    ]

    if dry_run:
        print(f"  [DRY RUN] train: {' '.join(cmd)}")
        return True

    log = RESULTS_DIR / "logs" / f"{name}_train.log"
    rc = run_subprocess(cmd, log, "TRAIN")
    if rc != 0:
        print(f"  ERROR: training failed (exit {rc}). See {log}")
        return False
    return True


# ---------------------------------------------------------------------------
# Core: evaluate one model
# ---------------------------------------------------------------------------

def evaluate(spec: dict, name: str, out_model_dir: Path, dry_run: bool) -> dict | None:
    station = spec.get("station", "ps117_39")
    station_cfg = STATIONS[station]
    iters = spec.get("iterations", 30000)
    out_json = RESULTS_DIR / "runs" / f"{name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PY), str(EVAL_PY),
        "--config", station_cfg["eval_config"],
        "--gs-root", str(GS_ROOT),
        "--model-path", str(out_model_dir),
        "--iteration", str(iters),
        "--out", str(out_json),
    ]

    if dry_run:
        print(f"  [DRY RUN] eval: {' '.join(cmd)}")
        return {"test_r_mean": None, "test_r_std": None,
                "test_rmse_mean": None, "test_rmse_std": None,
                "train_r_mean": None, "n_frames": None}

    log = RESULTS_DIR / "logs" / f"{name}_eval.log"
    rc = run_subprocess(cmd, log, "EVAL")
    if rc != 0:
        print(f"  ERROR: eval failed (exit {rc}). See {log}")
        return None

    with open(out_json) as f:
        d = json.load(f)

    # extract train r (mean over blocks)
    train_rs = [b["train"]["r"] for b in d.get("per_block", []) if b.get("train", {}).get("r") is not None]
    train_r_mean = sum(train_rs) / len(train_rs) if train_rs else None

    return {
        "test_r_mean": d.get("test_r_mean"),
        "test_r_std": d.get("test_r_std"),
        "test_rmse_mean": d.get("test_rmse_mean"),
        "test_rmse_std": d.get("test_rmse_std"),
        "train_r_mean": train_r_mean,
        "n_frames": d.get("n_frames"),
    }


# ---------------------------------------------------------------------------
# Run one experiment spec
# ---------------------------------------------------------------------------

def run_one(spec: dict, dry_run: bool):
    name = run_name(spec)
    out_model_dir = GS_ROOT / "output" / "pipeline" / name
    result_json = RESULTS_DIR / "runs" / f"{name}.json"

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  w={spec.get('sonar_loss_weight', 0.1)}  k={spec.get('sonar_adaptive_gain', 3.0)}"
          f"  iters={spec.get('iterations', 30000)}")

    # Skip if already done
    if result_json.exists() and not spec.get("force", False):
        print(f"  SKIP — result already exists at {result_json}")
        with open(result_json) as f:
            d = json.load(f)
        r_mean = d.get("test_r_mean")
        rmse_mean = d.get("test_rmse_mean")
        if r_mean is not None:
            print(f"  Cached: r={r_mean:+.3f}, RMSE={rmse_mean:.3f}m")
        return

    started = datetime.now().isoformat()
    t0 = time.time()

    # Train
    if not out_model_dir.exists() or spec.get("retrain", False):
        ok = train(spec, name, out_model_dir, dry_run)
        if not ok:
            return
    else:
        print(f"  Model already at {out_model_dir}, skipping training")

    # Eval
    metrics = evaluate(spec, name, out_model_dir, dry_run)
    if metrics is None:
        return

    elapsed = int(time.time() - t0)

    # Print result
    if metrics["test_r_mean"] is not None:
        print(f"  RESULT: r={metrics['test_r_mean']:+.3f}±{metrics['test_r_std']:.3f}"
              f"  RMSE={metrics['test_rmse_mean']:.3f}±{metrics['test_rmse_std']:.3f}m"
              f"  (train_r={metrics['train_r_mean']:+.3f})")

    # Append to CSV
    row = {
        "run_name": name,
        "station": spec.get("station", "ps117_39"),
        "iterations": spec.get("iterations", 30000),
        "sonar_loss_weight": spec.get("sonar_loss_weight", 0.1),
        "sonar_adaptive_gain": spec.get("sonar_adaptive_gain", 3.0),
        "dropout_pct": spec.get("dropout_pct", 0),
        "pingrate_decimation": spec.get("pingrate_decimation", 1),
        "started_at": started,
        "elapsed_s": elapsed,
        "notes": spec.get("notes", ""),
        **metrics,
    }
    if not dry_run:
        append_csv(row)
        print(f"  Appended to {CSV_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_yaml", help="Path to experiment YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running anything")
    args = parser.parse_args()

    spec = load_yaml(Path(args.experiment_yaml))
    runs = expand_sweep(spec)

    print(f"Loaded: {args.experiment_yaml}")
    print(f"Runs to execute: {len(runs)}")
    if args.dry_run:
        print("DRY RUN — no training or eval will occur\n")

    for i, run_spec in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}]")
        run_one(run_spec, dry_run=args.dry_run)

    # Print summary from CSV
    if not args.dry_run and CSV_PATH.exists():
        print(f"\n{'='*60}")
        print(f"Summary (all runs in {CSV_PATH}):")
        print(f"{'Run':<50} {'r':>7} {'RMSE':>7}")
        print("-" * 65)
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                r = row.get("test_r_mean", "")
                rmse = row.get("test_rmse_mean", "")
                try:
                    r_fmt = f"{float(r):+.3f}"
                    rmse_fmt = f"{float(rmse):.3f}m"
                except (ValueError, TypeError):
                    r_fmt, rmse_fmt = r, rmse
                print(f"  {row['run_name']:<48} {r_fmt:>7} {rmse_fmt:>7}")


if __name__ == "__main__":
    main()
