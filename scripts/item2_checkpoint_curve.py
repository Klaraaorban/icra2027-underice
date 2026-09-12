"""Item 2: evaluate PSNR + held-out sonar r at every saved checkpoint of a
densely-checkpointed training run, for the training-curve divergence plot.

For each checkpoint: renders all views once, computing (a) PSNR against each
view's own ground-truth image and (b) sonar r/RMSE using a scale fit on the
TRAIN split only (never touching test), applied to test-split frames -- same
train-only-fit discipline as everywhere else in this project.

Must run in the gaussian_splatting conda env. Run checkpoints SEQUENTIALLY,
one process per checkpoint if memory is tight (a single long-lived process
holding 10 checkpoints' Gaussians in memory back-to-back risks the same kind
of resource exhaustion seen when running this concongurrently with a live
training job -- here there's no concurrent job, but each checkpoint's Scene
load is still real memory, so this iterates with explicit cleanup between).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def evaluate_one_checkpoint(model_path, iteration, cfg):
    import torch
    import torch.nn.functional as F
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state
    import bisect
    from datetime import datetime, timedelta

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

    name_remap = None
    if cfg.get("index_to_orig_name"):
        with open(cfg["index_to_orig_name"]) as f:
            name_remap = json.load(f)

    _parser = _AP()
    model = ModelParams(_parser, sentinel=True)
    pipeline = PipelineParams(_parser)
    _parser.add_argument("--iteration", default=-1, type=int)
    sys.argv = ["x", "-m", str(model_path), "--iteration", str(iteration)]
    gs_args = get_combined_args(_parser)
    safe_state(False)

    gaussians = GaussianModel(model.extract(gs_args).sh_degree)
    scene = Scene(model.extract(gs_args), gaussians, load_iteration=gs_args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    entries = load_valeport(cfg["valeport_path"])
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(cfg["clip_start"])
    fps = cfg["fps"]

    def collect(views, split_name):
        out_rows = []
        for view in views:
            out = render(view, gaussians, pipeline.extract(gs_args), bg)
            rendered_img = out["render"].clamp(0, 1)
            gt_img = view.original_image.clamp(0, 1).to(rendered_img.device)
            mse = F.mse_loss(rendered_img, gt_img)
            frame_psnr = (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

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
            out_rows.append({
                "split": split_name, "psnr": frame_psnr,
                "sonar": ping, "pred_depth_units": center_depth_units,
            })
        return out_rows

    train_rows = collect(scene.getTrainCameras(), "train")
    test_rows = collect(scene.getTestCameras(), "test")

    all_psnr = [r["psnr"] for r in train_rows + test_rows]
    test_psnr = [r["psnr"] for r in test_rows]

    train_matched = [r for r in train_rows if r["sonar"] is not None and r["sonar"] != -99999]
    test_matched = [r for r in test_rows if r["sonar"] is not None and r["sonar"] != -99999]

    result = {"iteration": iteration, "train_psnr_mean": float(np.mean(train_rows and [r["psnr"] for r in train_rows] or [np.nan])),
              "test_psnr_mean": float(np.mean(test_psnr)) if test_psnr else None}

    if len(train_matched) >= 3:
        tr_units = np.array([r["pred_depth_units"] for r in train_matched])
        tr_sonar = np.array([r["sonar"] for r in train_matched])
        scale = float(np.sum(tr_sonar * tr_units) / max(np.sum(tr_units ** 2), 1e-9))
        result["fitted_scale"] = scale
        if len(test_matched) >= 3:
            te_units = np.array([r["pred_depth_units"] for r in test_matched])
            te_sonar = np.array([r["sonar"] for r in test_matched])
            pred_m = te_units * scale
            result["test_sonar_r"] = float(np.corrcoef(te_sonar, pred_m)[0, 1]) if np.std(pred_m) > 1e-9 else None
            result["test_sonar_rmse_m"] = float(np.sqrt(np.mean((te_sonar - pred_m) ** 2)))
        else:
            result["test_sonar_r"] = None
            result["test_sonar_rmse_m"] = None
    else:
        result["fitted_scale"] = None
        result["test_sonar_r"] = None
        result["test_sonar_rmse_m"] = None

    del scene, gaussians
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = []
    for it in args.iterations:
        print(f"=== evaluating iteration {it} ===")
        r = evaluate_one_checkpoint(args.model_path, it, cfg)
        print(json.dumps(r, indent=2))
        results.append(r)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
