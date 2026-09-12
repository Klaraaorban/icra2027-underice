"""Sonar-accuracy evaluation for an already-trained WaterSplatting (nerfstudio)
checkpoint. Same purpose and methodology as scripts/seasplat_sonar_eval.py:
this baseline has zero sonar supervision, so this measures its raw geometry
against real sonar the same way every other config in this project has been
scored (scale fit on TRAIN split only, scored on the model's own held-out
TEST split, report both Pearson r and RMSE).

Must run with an env that has both `nerfstudio` and `water_splatting`
installed (this project uses `gsplat_env`). No cwd requirement -- both are
installed packages, not local unpackaged modules like SeaSplat's fork.

Usage:
    python scripts/watersplatting_sonar_eval.py \
        --config ../icra2027_underice/configs/ps117_39.yaml \
        --ns-config "D:/watersplatting_output/ps117_39_vanilla_init/water-splatting/2026-08-27_164006/config.yml" \
        --out ../icra2027_underice/results/runs/watersplatting_ps117_39_sonar_eval.json
"""
import argparse
import bisect
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def load_simple_yaml(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.strip().strip('"').strip("'")
            cfg[key.strip()] = val
    return cfg


def load_valeport(path):
    entries = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append((datetime.fromisoformat(d["timestamp"]), float(d["Ping Depth"])))
            except Exception:
                pass
    entries.sort()
    return entries


def nearest_ping(entries, timestamps, ts, max_gap=1.0):
    i = bisect.bisect_left(timestamps, ts)
    cands = [j for j in (i - 1, i) if 0 <= j < len(timestamps)]
    best = min(cands, key=lambda j: abs((timestamps[j] - ts).total_seconds()))
    gap = abs((timestamps[best] - ts).total_seconds())
    return entries[best][1] if gap <= max_gap else None


def pearson_and_rmse_with_train_fit_scale(train_sonar, train_pred, eval_sonar, eval_pred):
    train_sonar = np.asarray(train_sonar, dtype=np.float64)
    train_pred = np.asarray(train_pred, dtype=np.float64)
    if len(train_sonar) < 3 or np.std(train_pred) < 1e-9:
        return None
    scale = float(np.sum(train_sonar * train_pred) / max(np.sum(train_pred ** 2), 1e-9))

    def score(sonar, pred):
        sonar = np.asarray(sonar, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if len(sonar) < 3 or np.std(pred) < 1e-9:
            return {"n": len(sonar), "r": None, "rmse": None}
        r = float(np.corrcoef(sonar, pred)[0, 1])
        resid = sonar - scale * pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        return {"n": int(len(sonar)), "r": r, "rmse": rmse}

    return {
        "fitted_scale": scale,
        "scale_fit_source": "train_split_only",
        "train": score(train_sonar, train_pred),
        "test": score(eval_sonar, eval_pred),
    }


def render_all_frame_depths_watersplatting(ns_config_path, valeport_path, clip_start_iso, fps,
                                            index_to_orig_name=None):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    name_remap = None
    if index_to_orig_name:
        with open(index_to_orig_name) as f:
            name_remap = json.load(f)

    config, pipeline, checkpoint_path, step = eval_setup(Path(ns_config_path))
    print(f"loaded checkpoint {checkpoint_path} at step {step}")
    pipeline.eval()

    entries = load_valeport(valeport_path)
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(clip_start_iso)

    rows = []
    with torch.no_grad():
        for split_name, dataset in [("train", pipeline.datamanager.train_dataset),
                                     ("test", pipeline.datamanager.eval_dataset)]:
            cameras = dataset.cameras
            image_filenames = dataset.image_filenames
            for i in range(len(image_filenames)):
                camera = cameras[i:i + 1]
                outputs = pipeline.model.get_outputs_for_camera(camera)
                depth = outputs["depth"]
                if depth.dim() == 3:
                    depth = depth[..., 0]
                h, w = depth.shape[-2], depth.shape[-1]
                cy, cx = h // 2, w // 2
                center_depth = depth[cy - 10:cy + 10, cx - 10:cx + 10].mean().item()

                name = Path(image_filenames[i]).stem
                orig_name = name_remap.get(name, name) if name_remap else name
                try:
                    frame_idx = int(orig_name.lstrip("f"))
                except ValueError:
                    continue
                t_s = frame_idx / fps
                est_ts = clip_start + timedelta(seconds=t_s)
                ping = nearest_ping(entries, timestamps, est_ts)
                if ping is not None and ping != -99999:
                    rows.append({"orig_split": split_name, "name": name, "orig_name": orig_name,
                                 "t_s": t_s, "sonar": ping, "pred_depth": center_depth})

    rows.sort(key=lambda r: r["t_s"])
    return rows, step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="project sensor config (ps117_39.yaml)")
    parser.add_argument("--ns-config", required=True, help="nerfstudio run's config.yml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)

    rows, step = render_all_frame_depths_watersplatting(
        args.ns_config, cfg["valeport_path"], cfg["clip_start"], float(cfg["fps"]),
        cfg.get("index_to_orig_name"),
    )
    n = len(rows)
    train_rows = [r for r in rows if r["orig_split"] == "train"]
    test_rows = [r for r in rows if r["orig_split"] == "test"]

    result = pearson_and_rmse_with_train_fit_scale(
        [r["sonar"] for r in train_rows], [r["pred_depth"] for r in train_rows],
        [r["sonar"] for r in test_rows], [r["pred_depth"] for r in test_rows],
    )

    summary = {
        "model": "WaterSplatting (nerfstudio, no sonar supervision -- raw scattering-model geometry)",
        "ns_config": str(args.ns_config),
        "step": int(step),
        "config": args.config,
        "n_frames_matched": n,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "holdout_method": "watersplatting's own ColmapDataParser eval_mode=interval split "
                           "(checkpoint was trained this way; not re-split to keep this an honest "
                           "reflection of the delivered checkpoint)",
        "result": result,
        "generated_at": datetime.now().isoformat(),
    }
    print(f"n={n} (train={len(train_rows)}, test={len(test_rows)})")
    if result:
        print(f"train: r={result['train']['r']}, rmse={result['train']['rmse']}")
        print(f"test:  r={result['test']['r']}, rmse={result['test']['rmse']}")
    else:
        print("FAILED: insufficient train data to fit scale")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")

    csv_path = Path(args.out).with_suffix(".rows.csv")
    with open(csv_path, "w") as f:
        f.write("orig_split,name,orig_name,t_s,sonar,pred_depth\n")
        for r in rows:
            f.write(f"{r['orig_split']},{r['name']},{r['orig_name']},{r['t_s']},{r['sonar']},{r['pred_depth']}\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
