"""Generate a dense sonar trace from a trained 3DGS model.

The altimeter pings at ~3.92 Hz (one ping every ~6 frames at 25 fps). The 3DGS
reconstruction can provide a rendered depth estimate at every video frame,
giving a dense ice-distance trace at 25 fps. This is the downstream use of
the better geometry our sonar-supervised model provides: a ~6x denser depth
profile of the ice underside, interpolated from the sparse altimeter signal.

Outputs a JSON file with per-frame timestamps, rendered depths (converted to
metres using the fitted scale), and real pings where available.

Usage:
    python gen_dense_trace.py \
        --config configs/ps117_39.yaml \
        --gs-root /path/to/gaussian-splatting \
        --model-path output/ps117_39_da3_texconf_calibrated_sonar_30k \
        --iteration 30000 \
        --scale 5.2421 \
        --out dense_trace_ours.json

    --scale: the fitted sonar_scale (from sonar_scale.json in the model output,
             or from eval_ours.py's reported per-block scale).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sonar_utils import load_valeport, nearest_ping, frame_timestamp, load_config


def render_all_frames(gs_root, model_path, iteration, half_patch=10, index_to_orig_name=None):
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
                rows.append({"split": split, "name": orig_name, "pred_depth_colmap": pred})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gs-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--scale", type=float, required=True,
                        help="Fitted sonar scale (COLMAP units → metres). From sonar_scale.json "
                             "or eval_ours.py output.")
    parser.add_argument("--patch-size", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    clip_start = datetime.fromisoformat(cfg["clip_start"])
    fps = float(cfg["fps"])

    index_to_orig_name = None
    if "index_to_orig_name" in cfg:
        with open(cfg["index_to_orig_name"]) as f:
            index_to_orig_name = json.load(f)

    # Load valeport for real ping matching
    entries = load_valeport(cfg["valeport_path"])
    timestamps = [e[0] for e in entries]

    # Render all frames
    rows = render_all_frames(args.gs_root, args.model_path, args.iteration,
                             half_patch=args.patch_size // 2,
                             index_to_orig_name=index_to_orig_name)

    # Add timestamps and sonar matches
    output = []
    for row in rows:
        try:
            est_ts, t_s = frame_timestamp(row["name"], fps, clip_start)
        except (ValueError, TypeError):
            continue
        pred_m = row["pred_depth_colmap"] * args.scale
        ping, gap_s = nearest_ping(entries, timestamps, est_ts)
        sonar_m = ping if (ping is not None and ping != -99999) else None
        output.append({
            "name": row["name"],
            "t_s": t_s,
            "split": row["split"],
            "rendered_depth_m": pred_m,
            "sonar_depth_m": sonar_m,
            "ping_gap_s": gap_s,
        })

    output.sort(key=lambda r: r["t_s"])

    n_with_ping = sum(1 for r in output if r["sonar_depth_m"] is not None)
    print(f"Total frames rendered: {len(output)}")
    print(f"Frames with a real ping (gap ≤ 1.0 s): {n_with_ping}")
    print(f"Dense interpolated frames: {len(output) - n_with_ping}")
    print(f"Effective depth-trace rate: {fps:.1f} Hz (vs altimeter {n_with_ping / (output[-1]['t_s'] - output[0]['t_s']):.2f} Hz)")

    # Summary statistics
    matched_sonar = [r["sonar_depth_m"] for r in output
                     if r["sonar_depth_m"] is not None and r["sonar_depth_m"] != -99999]
    matched_pred = [r["rendered_depth_m"] for r in output
                    if r["sonar_depth_m"] is not None and r["sonar_depth_m"] != -99999]
    if len(matched_sonar) >= 3:
        r = float(np.corrcoef(matched_sonar, matched_pred)[0, 1])
        rmse = float(np.sqrt(np.mean((np.array(matched_sonar) - np.array(matched_pred)) ** 2)))
        print(f"At ping frames — r = {r:+.3f}, RMSE = {rmse:.3f} m (scale = {args.scale:.4f})")

    with open(args.out, "w") as f:
        json.dump({"scale": args.scale, "fps": fps,
                   "n_total": len(output), "n_with_ping": n_with_ping,
                   "frames": output}, f, indent=2)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
