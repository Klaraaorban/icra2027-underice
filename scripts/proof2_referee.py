"""Proof 2: is the per-frame altimeter residual a usable reliability signal,
where PSNR is not? Computes, for every frame of a trained model, in one pass:
  - |residual| = |rendered_depth_m - sonar_depth_m|  (train-fit scale, per
    acoustic_holdout's convention -- not fit on that frame's own test status)
  - per-frame PSNR against that frame's own ground-truth image
so both signals can be directly compared along the same track, frame by frame,
without the render.py/metrics.py per_view.json index-vs-original-name mismatch
(per_view.json keys renders by sequential position, not by original frame
name, and only covers whichever split was rendered -- this script covers ALL
frames, train+test, keyed by real frame name, in one shared pass).

Must run in the gaussian_splatting conda env.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import bisect

import numpy as np
import torch
import torch.nn.functional as F
import yaml

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


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


def psnr_torch(rendered, gt):
    mse = F.mse_loss(rendered, gt)
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state
    import time as _time

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name_remap = None
    if cfg.get("index_to_orig_name"):
        with open(cfg["index_to_orig_name"]) as f:
            name_remap = json.load(f)

    _parser = _AP()
    model = ModelParams(_parser, sentinel=True)
    pipeline = PipelineParams(_parser)
    _parser.add_argument("--iteration", default=-1, type=int)
    sys.argv = ["x", "-m", str(args.model_path), "--iteration", str(args.iteration)]
    gs_args = get_combined_args(_parser)
    safe_state(False)

    gaussians = GaussianModel(model.extract(gs_args).sh_degree)
    scene = Scene(model.extract(gs_args), gaussians, load_iteration=gs_args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    entries = load_valeport(cfg["valeport_path"])
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(cfg["clip_start"])
    fps = cfg["fps"]

    # first, fit scale on TRAIN split only (rule 4)
    train_rows = []
    for view in scene.getTrainCameras():
        t0 = _time.time()
        out = render(view, gaussians, pipeline.extract(gs_args), bg)
        render_ms = (_time.time() - t0) * 1000
        depth = out["depth"]
        _, h, w = depth.shape
        cy, cx = h // 2, w // 2
        center_depth_units = depth[0, cy - 10:cy + 10, cx - 10:cx + 10].mean().item()
        name = Path(view.image_name).stem
        orig_name = name_remap.get(name, name) if name_remap else name
        try:
            frame_idx = int(orig_name.lstrip("f"))
        except ValueError:
            continue
        t_s = frame_idx / fps
        est_ts = clip_start + timedelta(seconds=t_s)
        ping = nearest_ping(entries, timestamps, est_ts)
        if ping is not None and ping != -99999:
            train_rows.append((center_depth_units, ping))
    tr_units = np.array([r[0] for r in train_rows])
    tr_sonar = np.array([r[1] for r in train_rows])
    scale = float(np.sum(tr_sonar * tr_units) / max(np.sum(tr_units ** 2), 1e-9))
    print(f"train-fit scale: {scale:.4f} (n_train={len(train_rows)})")

    rows = []
    for split_name, views in [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]:
        for view in views:
            t0 = _time.time()
            out = render(view, gaussians, pipeline.extract(gs_args), bg)
            render_ms = (_time.time() - t0) * 1000
            rendered_img = out["render"].clamp(0, 1)
            gt_img = view.original_image.clamp(0, 1).to(rendered_img.device)
            frame_psnr = psnr_torch(rendered_img, gt_img)

            depth = out["depth"]
            _, h, w = depth.shape
            cy, cx = h // 2, w // 2
            center_depth_units = depth[0, cy - 10:cy + 10, cx - 10:cx + 10].mean().item()
            pred_m = center_depth_units * scale

            name = Path(view.image_name).stem
            orig_name = name_remap.get(name, name) if name_remap else name
            try:
                frame_idx = int(orig_name.lstrip("f"))
            except ValueError:
                continue
            t_s = frame_idx / fps
            est_ts = clip_start + timedelta(seconds=t_s)
            ping = nearest_ping(entries, timestamps, est_ts)
            if ping is None or ping == -99999:
                continue
            residual_m = pred_m - ping

            rows.append({
                "name": orig_name, "split": split_name, "t_s": t_s,
                "sonar_m": ping, "pred_m": pred_m, "residual_m": residual_m,
                "abs_residual_m": abs(residual_m), "psnr_db": frame_psnr,
                "render_ms": render_ms,
            })

    rows.sort(key=lambda r: r["t_s"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(args.out_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    abs_res = np.array([r["abs_residual_m"] for r in rows])
    psnrs = np.array([r["psnr_db"] for r in rows])
    render_times = np.array([r["render_ms"] for r in rows])
    print(f"n frames: {len(rows)}")
    print(f"mean |residual|: {abs_res.mean():.3f} m, mean PSNR: {psnrs.mean():.2f} dB")
    print(f"mean render time: {render_times.mean():.2f} ms/frame")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
