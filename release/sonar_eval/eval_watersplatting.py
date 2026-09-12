"""Evaluate a trained WaterSplatting (nerfstudio) checkpoint against the
real altimeter trace. Same purpose as eval_seasplat.py: no sonar
supervision, raw geometry scored on the checkpoint's own train/test split.

Requires an env with `nerfstudio` and `water_splatting` installed.
On Windows, set PYTHONUTF8=1 first -- nerfstudio prints an emoji that
crashes on the default console codepage otherwise.

    python eval_watersplatting.py --config configs/ps117_39.yaml \
        --ns-config /path/to/run/config.yml --out results_watersplatting.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from sonar_utils import (load_valeport, nearest_ping, frame_timestamp,
                         fit_scale_and_score, load_config, contiguous_block_split)


def render_frame_depths(ns_config_path, valeport_path, clip_start_iso, fps, half_patch=10):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, checkpoint_path, step = eval_setup(Path(ns_config_path))
    pipeline.eval()

    entries = load_valeport(valeport_path)
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(clip_start_iso)

    rows = []
    with torch.no_grad():
        for split, dataset in [("train", pipeline.datamanager.train_dataset),
                                ("test", pipeline.datamanager.eval_dataset)]:
            cameras = dataset.cameras
            image_filenames = dataset.image_filenames
            for i in range(len(image_filenames)):
                depth = pipeline.model.get_outputs_for_camera(cameras[i:i + 1])["depth"]
                if depth.dim() == 3:
                    depth = depth[..., 0]
                h, w = depth.shape[-2], depth.shape[-1]
                cy, cx = h // 2, w // 2
                pred = depth[cy - half_patch:cy + half_patch, cx - half_patch:cx + half_patch].mean().item()

                orig_name = Path(image_filenames[i]).stem
                try:
                    est_ts, t_s = frame_timestamp(orig_name, fps, clip_start)
                except ValueError:
                    continue
                ping, gap_s = nearest_ping(entries, timestamps, est_ts)
                if ping is not None and ping != -99999:
                    rows.append({"split": split, "name": orig_name, "t_s": t_s,
                                 "sonar": ping, "pred_depth": pred, "gap_s": gap_s})

    rows.sort(key=lambda r: r["t_s"])
    return rows, step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ns-config", required=True)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=20,
                        help="Side length of the center patch used to sample depth (px). Default 20.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows, step = render_frame_depths(args.ns_config, cfg["valeport_path"], cfg["clip_start"],
                                     float(cfg["fps"]), half_patch=args.patch_size // 2)

    sonar = np.array([r["sonar"] for r in rows])
    pred = np.array([r["pred_depth"] for r in rows])

    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    per_block = []
    for block in range(args.n_blocks):
        train_idx, test_idx = contiguous_block_split(len(rows), args.n_blocks, block)
        result = fit_scale_and_score(sonar[train_idx], pred[train_idx], sonar[test_idx], pred[test_idx])
        if result is None:
            print(f"block={block}  skipped (degenerate split)")
            continue
        result["block"] = block
        per_block.append(result)
        print(f"block={block}  scale={fmt(result['scale'])}  "
              f"train r={fmt(result['train']['r'])} rmse={fmt(result['train']['rmse'])}  "
              f"test r={fmt(result['test']['r'])} rmse={fmt(result['test']['rmse'])} "
              f"rmse_unscaled={fmt(result['test']['rmse_unscaled'])}")

    test_rs = [b["test"]["r"] for b in per_block if b["test"]["r"] is not None]
    test_rmses = [b["test"]["rmse"] for b in per_block if b["test"]["rmse"] is not None]
    test_rmses_unscaled = [b["test"]["rmse_unscaled"] for b in per_block if b["test"]["rmse_unscaled"] is not None]
    gaps = [r["gap_s"] for r in rows]
    summary = {
        "model": "WaterSplatting (no sonar supervision)",
        "ns_config": str(args.ns_config),
        "step": int(step),
        "patch_size": args.patch_size,
        "n_frames": len(rows),
        "n_blocks": args.n_blocks,
        "ping_gap_s": {"mean": float(np.mean(gaps)), "max": float(np.max(gaps)), "p95": float(np.percentile(gaps, 95))},
        "per_block": per_block,
        "test_r_mean": float(np.mean(test_rs)) if test_rs else None,
        "test_r_std": float(np.std(test_rs)) if test_rs else None,
        "test_rmse_mean": float(np.mean(test_rmses)) if test_rmses else None,
        "test_rmse_std": float(np.std(test_rmses)) if test_rmses else None,
        "test_rmse_unscaled_mean": float(np.mean(test_rmses_unscaled)) if test_rmses_unscaled else None,
        "generated_at": datetime.now().isoformat(),
    }
    if test_rs:
        print(f"\ntest r = {summary['test_r_mean']:.3f} +/- {summary['test_r_std']:.3f}, "
              f"test RMSE = {summary['test_rmse_mean']:.3f} +/- {summary['test_rmse_std']:.3f} m  "
              f"(unscaled: {summary['test_rmse_unscaled_mean']:.3f} m)")
        print(f"ping gap: mean={summary['ping_gap_s']['mean']:.3f}s  "
              f"p95={summary['ping_gap_s']['p95']:.3f}s  max={summary['ping_gap_s']['max']:.3f}s")
    else:
        print("\nno valid blocks to summarise")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
