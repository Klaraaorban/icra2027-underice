"""Lightweight PSNR-only extraction (no training, pure inference on an
already-trained checkpoint) for every view, keyed by REAL frame name --
avoids the per_view.json sequential-index-vs-original-name ambiguity by
rendering directly rather than reverse-engineering historical llffhold
orderings per config.

Must run in the gaussian_splatting conda env.
"""
import argparse
import csv
import sys
from pathlib import Path

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state

    _parser = _AP()
    model = ModelParams(_parser, sentinel=True)
    pipeline = PipelineParams(_parser)
    _parser.add_argument("--iteration", default=-1, type=int)
    # old checkpoints' saved cfg_args predate the texture_confidence/depths fields;
    # get_combined_args only overlays CLI args that are != None, so an old cfg_args
    # plus this script's own None defaults silently omits the field entirely unless
    # explicitly passed here (established fix, see project memory)
    sys.argv = ["x", "-m", str(args.model_path), "--iteration", str(args.iteration),
                "--texture_confidence", "", "--depths", ""]
    gs_args = get_combined_args(_parser)
    safe_state(False)

    gaussians = GaussianModel(model.extract(gs_args).sh_degree)
    scene = Scene(model.extract(gs_args), gaussians, load_iteration=gs_args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    rows = []
    for split_name, views in [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]:
        for view in views:
            out = render(view, gaussians, pipeline.extract(gs_args), bg)
            rendered_img = out["render"].clamp(0, 1)
            gt_img = view.original_image.clamp(0, 1).to(rendered_img.device)
            mse = F.mse_loss(rendered_img, gt_img)
            psnr = (20 * torch.log10(1.0 / torch.sqrt(mse))).item()
            rows.append({"name": Path(view.image_name).stem, "split": split_name, "psnr": psnr})

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "split", "psnr"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
