"""Evaluate our sonar-supervised 3DGS model against the real altimeter trace.

Holds out a contiguous along-track block of frames (not interleaved --
adjacent frames are too correlated for that to be a fair test), fits a
scale factor on the remaining frames only, then scores the held-out block.

Requires a trained checkpoint from the `gaussian-splatting` repo and its
conda env (needs the CUDA rasterizer).

    python eval_ours.py --config configs/ps117_39.yaml \
        --gs-root /path/to/gaussian-splatting \
        --model-path /path/to/gaussian-splatting/output/<run> \
        --iteration 30000 --out results_ours.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sonar_utils import (load_valeport, nearest_ping, frame_timestamp,
                         fit_scale_and_score, load_config, contiguous_block_split)


def render_frame_depths(gs_root, model_path, iteration, valeport_path, clip_start_iso, fps, half_patch=10,
                        index_to_orig_name=None):
    import torch
    sys.path.insert(0, str(gs_root))
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser
    from utils.general_utils import safe_state

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    sys.argv = ["x", "-m", str(model_path), "--iteration", str(iteration)]
    args = get_combined_args(parser)
    safe_state(False)

    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    entries = load_valeport(valeport_path)
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(clip_start_iso)

    rows = []
    with torch.no_grad():
        for split, views in [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]:
            for view in views:
                out = render(view, gaussians, pipeline.extract(args), bg)
                depth = out["depth"]
                _, h, w = depth.shape
                cy, cx = h // 2, w // 2
                pred = depth[0, cy - half_patch:cy + half_patch, cx - half_patch:cx + half_patch].mean().item()

                orig_name = Path(view.image_name).stem
                if index_to_orig_name is not None:
                    orig_name = index_to_orig_name.get(orig_name, orig_name)
                try:
                    est_ts, t_s = frame_timestamp(orig_name, fps, clip_start)
                except ValueError:
                    continue
                ping, gap_s = nearest_ping(entries, timestamps, est_ts)
                if ping is not None and ping != -99999:
                    rows.append({"split": split, "name": orig_name, "t_s": t_s,
                                 "sonar": ping, "pred_depth": pred, "gap_s": gap_s})

    rows.sort(key=lambda r: r["t_s"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gs-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=20,
                        help="Side length of the center patch used to sample depth (px). Default 20.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    index_to_orig_name = None
    if "index_to_orig_name" in cfg:
        with open(cfg["index_to_orig_name"]) as _f:
            index_to_orig_name = json.load(_f)
    rows = render_frame_depths(args.gs_root, args.model_path, args.iteration,
                                cfg["valeport_path"], cfg["clip_start"], float(cfg["fps"]),
                                half_patch=args.patch_size // 2,
                                index_to_orig_name=index_to_orig_name)

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
        "model_path": str(args.model_path),
        "iteration": args.iteration,
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
