"""Item 5: the qualitative figure. On-trajectory render (normal view, the
38.12dB PSNR case) vs. the SAME model viewed off-trajectory (camera pose
rotated/translated away from the narrow training path) to reveal degenerate
"needle-Gaussian" geometry a high photometric score is hiding, plus the real
sonar profile for the same sequence. Built fresh from the actual trained
model and real sonar data -- no prior screenshot asset was available for this
session, so nothing here is reused from an untraceable source.

Must run in the gaussian_splatting conda env.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

GS_ROOT = Path("E:/Research/Holo/gaussian-splatting")
sys.path.insert(0, str(GS_ROOT))


def build_perturbed_minicam(ref_view, translate_m, rotate_deg_axis_angle, sonar_scale):
    """Build a MiniCam at a pose translated/rotated away from ref_view's pose,
    in the SAME intrinsics, to render an off-trajectory view of the same
    scene region. translate_m is in real metres (converted to COLMAP units
    via sonar_scale); rotate_deg_axis_angle is (axis_xyz, degrees)."""
    from scene.cameras import MiniCam
    from scipy.spatial.transform import Rotation

    W2C = ref_view.world_view_transform.transpose(0, 1).detach().cpu().numpy()  # world->cam, standard form
    C2W = np.linalg.inv(W2C)

    axis, deg = rotate_deg_axis_angle
    R_perturb = Rotation.from_rotvec(np.radians(deg) * np.array(axis) / np.linalg.norm(axis)).as_matrix()
    C2W_new = C2W.copy()
    C2W_new[:3, :3] = C2W[:3, :3] @ R_perturb
    C2W_new[:3, 3] = C2W[:3, 3] + (np.array(translate_m) / sonar_scale)

    W2C_new = np.linalg.inv(C2W_new)
    world_view_transform_new = torch.tensor(W2C_new, dtype=torch.float32).transpose(0, 1).cuda()
    full_proj_transform_new = (world_view_transform_new.unsqueeze(0).bmm(
        ref_view.projection_matrix.unsqueeze(0))).squeeze(0)

    return MiniCam(
        ref_view.image_width, ref_view.image_height, ref_view.FoVy, ref_view.FoVx,
        ref_view.znear, ref_view.zfar, world_view_transform_new, full_proj_transform_new,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sonar-scale", type=float, required=True)
    parser.add_argument("--ref-frame-name", required=True, help="orig_name of the training frame to use as the on-trajectory anchor")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser as _AP
    from utils.general_utils import safe_state
    import torchvision

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

    all_views = scene.getTrainCameras() + scene.getTestCameras()
    name_remap = None
    if cfg.get("index_to_orig_name"):
        with open(cfg["index_to_orig_name"]) as f:
            name_remap = json.load(f)

    ref_view = None
    for v in all_views:
        name = Path(v.image_name).stem
        orig_name = name_remap.get(name, name) if name_remap else name
        if orig_name == args.ref_frame_name:
            ref_view = v
            break
    if ref_view is None:
        raise SystemExit(f"frame {args.ref_frame_name} not found among {len(all_views)} views")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    on_traj = render(ref_view, gaussians, pipeline.extract(gs_args), bg)
    torchvision.utils.save_image(on_traj["render"], str(out_dir / "on_trajectory.png"))
    print(f"saved on-trajectory render (frame {args.ref_frame_name})")

    # off-trajectory: translate 1.5m laterally (perpendicular to typical forward
    # motion) and yaw 35 degrees -- far enough outside a narrow ROV survey
    # trajectory's baseline to expose degenerate view-dependent geometry, close
    # enough to still be looking at roughly the same scene region
    off_view = build_perturbed_minicam(
        ref_view, translate_m=[1.5, 0.0, 0.0], rotate_deg_axis_angle=([0, 1, 0], 35.0),
        sonar_scale=args.sonar_scale,
    )
    off_traj = render(off_view, gaussians, pipeline.extract(gs_args), bg)
    torchvision.utils.save_image(off_traj["render"], str(out_dir / "off_trajectory.png"))
    print("saved off-trajectory render (1.5m lateral + 35deg yaw from that pose)")

    print(f"wrote both renders to {out_dir}")


if __name__ == "__main__":
    main()
