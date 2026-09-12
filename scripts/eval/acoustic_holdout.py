"""Reusable acoustic (single-beam altimeter) holdout evaluation module.

Re-scoped Task 1 from the ICRA 2027 project brief: the original spec holds out
contiguous SWATH BLOCKS (multibeam). This dataset has no multibeam -- only a
single along-track Valeport altimeter trace -- so this module holds out
contiguous ALONG-TRACK FRAME BLOCKS instead: the along-track analogue of the
same anti-leakage principle (adjacent frames along a slow ROV track are highly
spatially correlated, exactly like adjacent overlapping swaths would be).

This replaces the INTERLEAVED every-8th-frame llffhold convention used for
every sonar number reported earlier in this project -- that convention is a
weaker holdout (immediately-adjacent train frames, ~1.4s away, are highly
correlated with each interleaved test frame) and should not be relied on for
numbers that go in the paper. This module is the fix.

Every result reports BOTH Pearson r and RMSE (never one alone -- r is
scale-invariant, RMSE is not, and rule 3 of the brief requires both).

Must be run with the `gaussian_splatting` conda env (needs the CUDA rasterizer
to render each view's depth). Requires an already-trained model at --model-path.

Usage:
    python scripts/eval/acoustic_holdout.py \
        --config configs/ps117_39.yaml \
        --model-path E:/Research/Holo/gaussian-splatting/output/<model> \
        --iteration 30000 --n-blocks 5 --seed 0 \
        --out results/runs/ps117_39_30k_acoustic_holdout_seed0.json
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def contiguous_block_split(n_frames, n_blocks=5, test_block_idx=None, seed=0):
    """Split n_frames sequential frame indices into n_blocks contiguous blocks.
    Returns (train_idx, test_idx) -- test is ONE block, train is everything else.
    No test frame is adjacent-in-index to a train frame from across a block
    boundary the way interleaved-every-Nth leaves every test frame with train
    neighbours on both sides."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_frames, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    if test_block_idx is None:
        test_block_idx = int(rng.integers(0, n_blocks))
    test_idx = blocks[test_block_idx]
    train_idx = np.concatenate([b for i, b in enumerate(blocks) if i != test_block_idx])
    return np.sort(train_idx), np.sort(test_idx), test_block_idx


def pearson_and_rmse_with_train_fit_scale(train_sonar, train_pred, eval_sonar, eval_pred):
    """Fit the scale factor on TRAIN block only (rule 4: if the alignment step
    touches test data the metric is contaminated -- this explicitly does not),
    then apply that fixed scale to whichever set (train or test) is being scored."""
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
        "scale_fit_source": "train_block_only",
        "train": score(train_sonar, train_pred),
        "test": score(eval_sonar, eval_pred),
    }


def render_all_frame_depths(model_path, iteration, valeport_path, clip_start_iso, fps, index_to_orig_name=None):
    """Render every train+test view's center-patch depth and match it to the
    altimeter trace. Returns list of dicts sorted by along-track time."""
    import bisect
    import torch
    from datetime import timedelta
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state

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
    if index_to_orig_name:
        with open(index_to_orig_name) as f:
            name_remap = json.load(f)

    _parser = _AP()
    model = ModelParams(_parser, sentinel=True)
    pipeline = PipelineParams(_parser)
    _parser.add_argument("--iteration", default=-1, type=int)
    sys.argv = ["x", "-m", str(model_path), "--iteration", str(iteration)]
    args = get_combined_args(_parser)
    safe_state(False)

    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    entries = load_valeport(valeport_path)
    timestamps = [e[0] for e in entries]
    clip_start = datetime.fromisoformat(clip_start_iso)

    rows = []
    for split_name, views in [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]:
        for view in views:
            out = render(view, gaussians, pipeline.extract(args), bg)
            depth = out["depth"]
            _, h, w = depth.shape
            cy, cx = h // 2, w // 2
            center_depth = depth[0, cy - 10:cy + 10, cx - 10:cx + 10].mean().item()

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
                rows.append({"orig_split": split_name, "name": name, "orig_name": orig_name,
                             "t_s": t_s, "sonar": ping, "pred_depth": center_depth})

    rows.sort(key=lambda r: r["t_s"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Explicit list of contiguous test-block indices to hold out "
                              "(one run per entry). Default: every block 0..n_blocks-1 "
                              "(full leave-one-block-out CV, strictly >= rule 3's 3-seed minimum "
                              "for n_blocks>=3, and avoids the random-seed collision risk of "
                              "sampling with replacement from a small number of blocks).")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seeds is None:
        args.seeds = list(range(args.n_blocks))

    frame_rows = render_all_frame_depths(
        args.model_path, args.iteration, cfg["valeport_path"], cfg["clip_start"], cfg["fps"],
        cfg.get("index_to_orig_name"),
    )
    n = len(frame_rows)
    sonar_all = np.array([r["sonar"] for r in frame_rows])
    pred_all = np.array([r["pred_depth"] for r in frame_rows])

    per_seed_results = []
    for seed in args.seeds:
        train_idx, test_idx, test_block_idx = contiguous_block_split(n, args.n_blocks, test_block_idx=seed)
        result = pearson_and_rmse_with_train_fit_scale(
            sonar_all[train_idx], pred_all[train_idx], sonar_all[test_idx], pred_all[test_idx],
        )
        result["seed"] = seed
        result["test_block_idx"] = test_block_idx
        result["n_blocks"] = args.n_blocks
        # diagnostic: is this block's sonar range narrower than the full sequence's?
        # (checks whether a contiguous-block RMSE improvement is real or just a
        # narrower/easier local range being scored)
        result["test_block_sonar_range_m"] = float(sonar_all[test_idx].max() - sonar_all[test_idx].min())
        per_seed_results.append(result)
        print(f"seed={seed} test_block={test_block_idx} (sonar range {result['test_block_sonar_range_m']:.3f}m): "
              f"train r={result['train']['r']:.3f} rmse={result['train']['rmse']:.3f} | "
              f"test r={result['test']['r']:.3f} rmse={result['test']['rmse']:.3f}")

    test_rs = [r["test"]["r"] for r in per_seed_results if r["test"]["r"] is not None]
    test_rmses = [r["test"]["rmse"] for r in per_seed_results if r["test"]["rmse"] is not None]
    summary = {
        "model_path": str(args.model_path),
        "iteration": args.iteration,
        "config": args.config,
        "n_frames_matched": n,
        "n_blocks": args.n_blocks,
        "holdout_method": "contiguous_block (Task 1 re-scoped for single-beam altimeter)",
        "scale_fit_source": "train_block_only",
        "per_seed": per_seed_results,
        "test_r_mean": float(np.mean(test_rs)) if test_rs else None,
        "test_r_std": float(np.std(test_rs)) if test_rs else None,
        "test_rmse_mean": float(np.mean(test_rmses)) if test_rmses else None,
        "test_rmse_std": float(np.std(test_rmses)) if test_rmses else None,
        "full_sequence_sonar_range_m": float(sonar_all.max() - sonar_all.min()),
        "generated_at": datetime.now().isoformat(),
    }
    print(f"\nSUMMARY (contiguous-block holdout, {len(args.seeds)} seeds): "
          f"test r = {summary['test_r_mean']:.3f} +/- {summary['test_r_std']:.3f}, "
          f"test RMSE = {summary['test_rmse_mean']:.3f} +/- {summary['test_rmse_std']:.3f} m")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
