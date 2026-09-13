"""Extract per-frame (sonar, rendered_depth) pairs from the SeaSplat checkpoint.

The SeaSplat checkpoint was trained before we added 'depths', 'train_test_exp',
and 'texture_confidence' to ModelParams, so its cfg_args lacks those keys.
This script patches the loaded args with sensible defaults before calling Scene.

Run from the gaussian-splatting root with the gaussian_splatting conda env:

    python extract_seasplat_pairs.py
"""
import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

EVAL_DIR   = Path(r"E:\Research\Holo\icra2027_underice\release\sonar_eval")
GS_ROOT    = Path(r"E:\Research\Holo\gaussian-splatting")
MODEL_PATH = Path(r"E:\Research\Holo\gaussian-splatting\data\ps117_39_30to90_fisheye\experiments\08272026\test")
ITERATION  = 7000
VALEPORT   = r"F:/AAS_4408_PS117_ROV_sample/PS117-39/valeport.dat"
CLIP_START = "2019-01-10T12:43:48"
FPS        = 25.0
HALF_PATCH = 10
OUT_PAIRS  = EVAL_DIR / "seasplat_pairs.json"

sys.path.insert(0, str(GS_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from sonar_utils import load_valeport, nearest_ping, frame_timestamp

import torch
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from utils.general_utils import safe_state

parser = ArgumentParser()
model  = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
sys.argv = ["x", "-m", str(MODEL_PATH), "--iteration", str(ITERATION)]
args = get_combined_args(parser)

# Patch attributes added after the SeaSplat checkpoint was trained
for attr, default in [("depths", ""), ("texture_confidence", ""), ("train_test_exp", False)]:
    if not hasattr(args, attr):
        setattr(args, attr, default)

safe_state(False)
gaussians = GaussianModel(model.extract(args).sh_degree)
scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
bg = bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

entries = load_valeport(VALEPORT)
timestamps = [e[0] for e in entries]
clip_start = datetime.fromisoformat(CLIP_START)

bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
bg_white = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
pp = pipeline.extract(args)

rows = []
with torch.no_grad():
    for split, views in [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]:
        for view in views:
            out_b = render(view, gaussians, pp, bg_black)
            out_w = render(view, gaussians, pp, bg_white)
            depth = out_b["depth"]
            # alpha from two-background trick: alpha = 1 - (render_white - render_black)
            alpha = 1.0 - (out_w["render"] - out_b["render"]).mean(dim=0, keepdim=True).clamp(0, 1)
            alpha = alpha.clamp(min=1e-6)
            # alpha-normalised depth, matching SeaSplat's render_depth / alpha approach
            depth_norm = depth / alpha
            _, h, w = depth_norm.shape
            cy, cx = h // 2, w // 2
            pred = depth_norm[0, cy - HALF_PATCH:cy + HALF_PATCH, cx - HALF_PATCH:cx + HALF_PATCH].mean().item()

            orig_name = Path(view.image_name).stem
            try:
                est_ts, t_s = frame_timestamp(orig_name, FPS, clip_start)
            except ValueError:
                continue
            ping, gap_s = nearest_ping(entries, timestamps, est_ts)
            if ping is not None and ping != -99999:
                rows.append({"name": orig_name, "t_s": t_s, "split": split,
                             "sonar_m": ping, "pred_depth": pred})

rows.sort(key=lambda r: r["t_s"])
OUT_PAIRS.write_text(json.dumps(rows, indent=2))
print(f"Saved {len(rows)} pairs to {OUT_PAIRS}")
# quick sanity: per-fold r
from sonar_utils import contiguous_block_split, fit_scale_and_score
import numpy as np
sonar_a = np.array([r["sonar_m"] for r in rows])
pred_a  = np.array([r["pred_depth"] for r in rows])
rs = []
for b in range(5):
    tr, te = contiguous_block_split(len(rows), 5, b)
    res = fit_scale_and_score(sonar_a[tr], pred_a[tr], sonar_a[te], pred_a[te])
    if res: rs.append(res["test"]["r"])
print(f"Per-fold test r: {[round(x,3) for x in rs]}  mean={np.mean(rs):+.3f}")
