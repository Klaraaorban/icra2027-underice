"""Proof 1 evaluation: score a trained model against its seed's FROZEN test
set (written by proof1_decimate_sonar.py, identical across every decimation
condition for that seed). Fits scale on the model's own train pool (whatever
sonar it was actually supervised with for that condition) and applies that
fixed scale to the frozen test set -- never fit on test.

Must run in the gaussian_splatting conda env.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
from acoustic_holdout import render_all_frame_depths  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--test-names-json", required=True)
    parser.add_argument("--train-sonar-depths-json", required=True,
                         help="The condition's own decimated sonar_depths.json -- used ONLY to "
                              "fit the scale factor on whatever pings that condition actually saw")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.test_names_json) as f:
        test_names = set(json.load(f))
    with open(args.train_sonar_depths_json) as f:
        train_sonar = json.load(f)

    all_rows, _ = render_all_frames_with_pose_stub(args, cfg)

    train_rows = [r for r in all_rows if r["orig_name"] in train_sonar]
    if len(train_rows) >= 3:
        tr_units = np.array([r["pred_depth"] for r in train_rows])
        tr_sonar = np.array([r["sonar"] for r in train_rows])
        scale = float(np.sum(tr_sonar * tr_units) / max(np.sum(tr_units ** 2), 1e-9))
    else:
        scale = None  # "none" condition -- no train sonar at all to fit a scale from

    test_rows = [r for r in all_rows if r["orig_name"] in test_names]
    result = {
        "station": args.station, "seed": args.seed, "condition": args.condition,
        "model_path": str(args.model_path), "n_train_sonar_pings": len(train_sonar),
        "n_test_matched": len(test_rows), "fitted_scale": scale,
    }
    if scale is not None and len(test_rows) >= 3:
        te_units = np.array([r["pred_depth"] for r in test_rows])
        te_sonar = np.array([r["sonar"] for r in test_rows])
        pred_m = te_units * scale
        r = float(np.corrcoef(te_sonar, pred_m)[0, 1]) if np.std(pred_m) > 1e-9 else None
        rmse = float(np.sqrt(np.mean((te_sonar - pred_m) ** 2)))
        result["test_r"] = r
        result["test_rmse_m"] = rmse
    else:
        result["test_r"] = None
        result["test_rmse_m"] = None
        result["note"] = ("no scale fit possible (condition=none has zero train sonar pings) -- "
                           "test_r/rmse intentionally null, not fabricated as 0 or skipped silently")

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def render_all_frames_with_pose_stub(args, cfg):
    # thin wrapper: acoustic_holdout's render_all_frame_depths already returns
    # (name, orig_name, t_s, sonar_m, pred_depth_units) for every train+test view
    rows = render_all_frame_depths(
        args.model_path, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"),
    )
    return rows, None


if __name__ == "__main__":
    main()
