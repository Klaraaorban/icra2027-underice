"""Shared helpers: load the altimeter trace, match it to frames by
timestamp, and score predicted depth against it."""
import bisect
import json
from datetime import datetime, timedelta

import numpy as np


def load_valeport(path):
    entries = []
    skipped = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append((datetime.fromisoformat(d["timestamp"]), float(d["Ping Depth"])))
            except Exception:
                skipped += 1
    if skipped:
        print(f"load_valeport: skipped {skipped} malformed line(s) in {path}")
    entries.sort()
    return entries


def nearest_ping(entries, timestamps, ts, max_gap=1.0):
    """Return (depth, gap_seconds) for the nearest ping, or (None, None) if beyond max_gap."""
    i = bisect.bisect_left(timestamps, ts)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(timestamps)]
    best = min(candidates, key=lambda j: abs((timestamps[j] - ts).total_seconds()))
    gap = abs((timestamps[best] - ts).total_seconds())
    if gap <= max_gap:
        return entries[best][1], gap
    return None, None


def frame_timestamp(orig_name, fps, clip_start):
    frame_idx = int(orig_name.lstrip("f"))
    t_s = frame_idx / fps
    return clip_start + timedelta(seconds=t_s), t_s


def fit_scale_and_score(train_sonar, train_pred, eval_sonar, eval_pred):
    """Fit sonar ~= scale * pred on TRAIN only, then score both splits
    against that fixed scale. Test data never touches the fit."""
    train_sonar = np.asarray(train_sonar, dtype=np.float64)
    train_pred = np.asarray(train_pred, dtype=np.float64)
    if len(train_sonar) < 3 or np.std(train_pred) < 1e-9:
        return None
    scale = float(np.sum(train_sonar * train_pred) / max(np.sum(train_pred ** 2), 1e-9))

    def score(sonar, pred):
        sonar = np.asarray(sonar, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if len(sonar) < 3 or np.std(pred) < 1e-9:
            return {"n": len(sonar), "r": None, "rmse": None, "rmse_unscaled": None}
        r = float(np.corrcoef(sonar, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((sonar - scale * pred) ** 2)))
        rmse_unscaled = float(np.sqrt(np.mean((sonar - pred) ** 2)))
        return {"n": int(len(sonar)), "r": r, "rmse": rmse, "rmse_unscaled": rmse_unscaled}

    return {"scale": scale, "train": score(train_sonar, train_pred), "test": score(eval_sonar, eval_pred)}


def contiguous_block_split(n_frames, n_blocks, test_block_idx):
    """Split frame indices into train/test as one contiguous along-track block."""
    edges = np.linspace(0, n_frames, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    test_idx = blocks[test_block_idx]
    train_idx = np.concatenate([b for i, b in enumerate(blocks) if i != test_block_idx])
    return np.sort(train_idx), np.sort(test_idx)


def load_config(path):
    # Minimal flat-key parser. Splits on the first colon only (partition),
    # so values containing colons (timestamps, Windows paths) are preserved.
    # Does not handle multi-line values, lists, or nested YAML structures.
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            cfg[key.strip()] = val.strip().strip('"').strip("'")
    return cfg
