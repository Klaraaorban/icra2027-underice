"""Assemble the qualitative figure: on-trajectory render, off-trajectory
render (revealing degenerate geometry), and the real sonar profile for this
sequence, in one composite. Pure matplotlib/PIL, no GPU needed -- run this
after item5_qualitative_figure.py has produced the two PNGs.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--sonar-csv", required=True, help="CSV with t_s,sonar columns (e.g. ps29_sonar_diag.csv)")
    parser.add_argument("--ref-t-s", type=float, required=True, help="t_s of the on-trajectory anchor frame, marked on the profile")
    parser.add_argument("--psnr-label", default="")
    parser.add_argument("--out-png", required=True)
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    on_img = Image.open(render_dir / "on_trajectory.png")
    off_img = Image.open(render_dir / "off_trajectory.png")

    t_vals, sonar_vals = [], []
    with open(args.sonar_csv) as f:
        for row in csv.DictReader(f):
            t_vals.append(float(row["t_s"]))
            sonar_vals.append(float(row["sonar"]))

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.35, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(on_img)
    ax1.set_title(f"On-trajectory render{(' — ' + args.psnr_label) if args.psnr_label else ''}", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(off_img)
    ax2.set_title("Same model, off-trajectory (1.5m lateral + 35° yaw)", fontsize=12)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_vals, sonar_vals, color="#1f6feb", linewidth=1.5)
    ax3.axvline(args.ref_t_s, color="#d1242f", linestyle="--", linewidth=1.2, label="on-trajectory frame")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("altimeter range (m)")
    ax3.set_title("Real sonar profile for this sequence", fontsize=11)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
