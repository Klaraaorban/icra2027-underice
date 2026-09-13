"""Generate Figure 2: scatter of rendered depth vs altimeter range for all methods.

Uses the same 5-fold contiguous along-track protocol as eval_ours.py:
  - Scale is fitted on training frames per fold, applied to that fold's test frames.
  - All 5 × ~60 held-out test points are shown in the scatter.
  - Pearson r is reported on the aggregate held-out set (scale-invariant, so
    per-fold scale differences do not affect the r annotation).

Output: fig2_signflip.pdf + fig2_signflip.png (300 dpi, column-width 8.5 cm).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

EVAL_DIR    = Path(__file__).parent
OURS_TRACE  = EVAL_DIR / "dense_trace_ours.json"
SPLAT_PAIRS = EVAL_DIR / "seasplat_pairs.json"
WATER_PAIRS = EVAL_DIR / "watersplatting_pairs.json"
OUT_PDF     = EVAL_DIR / "fig2_signflip.pdf"
OUT_PNG     = EVAL_DIR / "fig2_signflip.png"
N_BLOCKS    = 5

METHOD_CFG = [
    ("ours",           "Ours (sonar-sup., 30k)",  "#1a6faf", "o", 7,   0.85),
    ("seasplat",       "SeaSplat (7k)",            "#d62728", "s", 6,   0.75),
    ("watersplatting", "WaterSplatting",            "#ff7f0e", "^", 6,   0.75),
]

# ── helpers ------------------------------------------------------------------
def contiguous_block_split(n, n_blocks, test_block):
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    test_idx  = blocks[test_block]
    train_idx = np.concatenate([b for i, b in enumerate(blocks) if i != test_block])
    return np.sort(train_idx), np.sort(test_idx)

def cv_scaled_pairs(sonar_arr, pred_arr):
    """Return (x_test, y_test_scaled) aggregated over all 5 folds.

    Scale is fitted per fold on training data only (no-intercept LS).
    Pearson r on the returned arrays equals the scale-invariant per-fold mean r
    (up to the averaging vs. aggregation difference).
    """
    n = len(sonar_arr)
    all_x, all_y = [], []
    per_fold_r = []
    for b in range(N_BLOCKS):
        tr, te = contiguous_block_split(n, N_BLOCKS, b)
        s_tr, p_tr = sonar_arr[tr], pred_arr[tr]
        s_te, p_te = sonar_arr[te], pred_arr[te]
        # no-intercept scale: s = Σ(sonar·pred) / Σ(pred²)
        scale = float(np.dot(s_tr, p_tr) / max(np.dot(p_tr, p_tr), 1e-9))
        all_x.append(s_te)
        all_y.append(p_te * scale)
        if np.std(p_te) > 1e-9:
            per_fold_r.append(float(np.corrcoef(s_te, p_te)[0, 1]))
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    mean_r = float(np.mean(per_fold_r))
    std_r  = float(np.std(per_fold_r))
    return x, y, mean_r, std_r

# ── load data ----------------------------------------------------------------
def load_ours(path):
    """dense_trace: already in metres (scale pre-applied)."""
    d = json.loads(Path(path).read_text())
    rows = [f for f in d["frames"] if f.get("sonar_depth_m") is not None]
    rows.sort(key=lambda f: f["t_s"])
    sonar = np.array([f["sonar_depth_m"]    for f in rows])
    pred  = np.array([f["rendered_depth_m"] for f in rows])
    return sonar, pred

def load_pairs(path):
    """pairs: sonar_m in metres, pred_depth in COLMAP units (needs scaling)."""
    rows = sorted(json.loads(Path(path).read_text()), key=lambda r: r["t_s"])
    sonar = np.array([r["sonar_m"]    for r in rows])
    pred  = np.array([r["pred_depth"] for r in rows])
    return sonar, pred

# ── build datasets -----------------------------------------------------------
datasets = {}

if OURS_TRACE.exists():
    s, p = load_ours(OURS_TRACE)
    # Ours is already scaled; run CV just to get aggregate test pairs
    x, y, r_mean, r_std = cv_scaled_pairs(s, p)
    datasets["ours"] = (x, y, r_mean, r_std)
    print(f"Ours:          r={r_mean:+.3f}±{r_std:.3f}  n={len(x)}")
else:
    print(f"MISSING: {OURS_TRACE}")

if SPLAT_PAIRS.exists():
    s, p = load_pairs(SPLAT_PAIRS)
    x, y, r_mean, r_std = cv_scaled_pairs(s, p)
    datasets["seasplat"] = (x, y, r_mean, r_std)
    print(f"SeaSplat (7k): r={r_mean:+.3f}±{r_std:.3f}  n={len(x)}")
else:
    print(f"MISSING: {SPLAT_PAIRS}")

if WATER_PAIRS.exists():
    s, p = load_pairs(WATER_PAIRS)
    x, y, r_mean, r_std = cv_scaled_pairs(s, p)
    datasets["watersplatting"] = (x, y, r_mean, r_std)
    print(f"WaterSplatting:r={r_mean:+.3f}±{r_std:.3f}  n={len(x)}")
else:
    print(f"MISSING: {WATER_PAIRS}")

# ── plot ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5 / 2.54, 7.5 / 2.54))

legend_handles = []
for key, label, colour, marker, ms, alpha in METHOD_CFG:
    if key not in datasets:
        continue
    x, y, r_mean, r_std = datasets[key]
    slope, intercept, *_ = stats.linregress(x, y)

    ax.scatter(x, y, c=colour, marker=marker, s=ms**2, alpha=alpha,
               linewidths=0, rasterized=True, zorder=2)

    xl = np.linspace(x.min(), x.max(), 200)
    ax.plot(xl, slope * xl + intercept, color=colour, lw=1.5, zorder=3)

    patch = mpatches.Patch(color=colour,
                           label=f"{label}  $r={r_mean:+.2f}$")
    legend_handles.append(patch)

# ideal y = x line
all_x = np.concatenate([v[0] for v in datasets.values()])
all_y = np.concatenate([v[1] for v in datasets.values()])
lo = min(all_x.min(), all_y.min()) * 0.92
hi = max(all_x.max(), all_y.max()) * 1.05
ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=0.8, alpha=0.5, zorder=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

ax.set_xlabel("Altimeter range (m)", fontsize=8)
ax.set_ylabel("Rendered depth, fold-scaled (m)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.legend(handles=legend_handles, fontsize=6.5, loc="upper left",
          framealpha=0.9, edgecolor="0.8")
ax.set_aspect("equal", adjustable="box")

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"\nSaved {OUT_PDF}")
print(f"Saved {OUT_PNG}")
