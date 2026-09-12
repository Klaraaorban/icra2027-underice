"""Off-trajectory degradation, using REAL held-out frames rather than
synthetic perturbed poses (which would have no ground truth to score PSNR
against). For every test-split frame, computes its real 3D distance (metres)
to the nearest TRAINING camera position, then correlates that distance
against the frame's own real PSNR. A model with degenerate, "photo-collage"
geometry should show PSNR falling off sharply as distance-to-nearest-train-
view grows (since it never learned real 3D structure, just per-view color);
a model with genuinely consistent 3D geometry should degrade much more
gently, since nearby real geometry still projects correctly from a slightly
different vantage point.

Must run in the gaussian_splatting conda env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_test_1d import render_all_frames_with_pose  # noqa: E402

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sonar-scale", type=float, required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    import yaml
    import torch
    import torch.nn.functional as F
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    train_centers = np.array([v.camera_center.detach().cpu().numpy() for v in train_views]) * args.sonar_scale

    rows = []
    for view in test_views:
        out = render(view, gaussians, pipeline.extract(gs_args), bg)
        rendered_img = out["render"].clamp(0, 1)
        gt_img = view.original_image.clamp(0, 1).to(rendered_img.device)
        mse = F.mse_loss(rendered_img, gt_img)
        psnr = (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

        cam_center = view.camera_center.detach().cpu().numpy() * args.sonar_scale
        dists = np.linalg.norm(train_centers - cam_center, axis=1)
        nearest_dist = float(dists.min())

        rows.append({"name": Path(view.image_name).stem, "nearest_train_dist_m": nearest_dist, "psnr": psnr})

    df = pd.DataFrame(rows).sort_values("nearest_train_dist_m")
    df.to_csv(args.out_csv, index=False)
    print(df.to_string())

    rho, p = spearmanr(df["nearest_train_dist_m"], df["psnr"])
    print(f"\nn test frames: {len(df)}")
    print(f"Spearman rho(distance to nearest train view, PSNR): {rho:.3f}  (p={p:.4f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
