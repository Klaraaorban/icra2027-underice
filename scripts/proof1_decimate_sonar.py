"""Proof 1 setup: split first, decimate second (per the task's explicit order).

1. Sort all matched altimeter pings for a station by real frame number (temporal
   / along-track order -- sonar_depths.json keys directly encode frame number).
2. Apply the SAME contiguous-block holdout as Task 1's acoustic_holdout.py (test
   block IDENTICAL across every decimation condition for a given seed -- this is
   what makes the resulting r/RMSE curve comparable point to point).
3. Only THEN decimate inside the remaining TRAIN pool: full / every 2nd / every
   5th / every 10th / none (empty dict, i.e. no sonar supervision at all).

Writes one sonar_depths.json per (station, seed, condition) under
data/<station-dataset-dir>/proof1_decimated/, ready to pass straight to
train.py's --sonar_depths_json. Also writes the frozen TEST set (frame name
list) per (station, seed) so evaluation always scores the identical held-out
pings regardless of which decimation condition trained the model.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def contiguous_block_split_names(sorted_names, n_blocks, test_block_idx):
    n = len(sorted_names)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [sorted_names[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    test_names = blocks[test_block_idx]
    train_names = [nm for i, b in enumerate(blocks) if i != test_block_idx for nm in b]
    return train_names, test_names


def decimate(names, stride):
    if stride is None:  # "none" condition -- no sonar supervision at all
        return []
    return names[::stride]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonar-depths-json", required=True, help="Full matched sonar_depths.json for this station")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    with open(args.sonar_depths_json) as f:
        full = json.load(f)

    def frame_num(stem):
        return int(stem.lstrip("f")) if stem.lstrip("f").isdigit() else 0

    sorted_names = sorted(full.keys(), key=frame_num)
    print(f"n total matched pings: {len(sorted_names)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = {"full": 1, "every2nd": 2, "every5th": 5, "every10th": 10, "none": None}

    manifest = []
    for seed in args.seeds:
        train_names, test_names = contiguous_block_split_names(sorted_names, args.n_blocks, seed)
        print(f"seed={seed}: n_train_pool={len(train_names)}, n_test={len(test_names)}")

        test_path = out_dir / f"seed{seed}_test_names.json"
        with open(test_path, "w") as f:
            json.dump(test_names, f, indent=2)

        for cond_name, stride in conditions.items():
            decimated_names = decimate(train_names, stride)
            decimated_dict = {nm: full[nm] for nm in decimated_names}
            cond_path = out_dir / f"seed{seed}_{cond_name}_sonar_depths.json"
            with open(cond_path, "w") as f:
                json.dump(decimated_dict, f, indent=2)
            print(f"  {cond_name:10s}: n_sonar_supervised={len(decimated_dict)}  -> {cond_path.name}")
            manifest.append({
                "seed": seed, "condition": cond_name, "n_sonar_supervised": len(decimated_dict),
                "n_test_held_out": len(test_names), "sonar_depths_path": str(cond_path),
                "test_names_path": str(test_path),
            })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest.json ({len(manifest)} run configs) to {out_dir}")


if __name__ == "__main__":
    main()
