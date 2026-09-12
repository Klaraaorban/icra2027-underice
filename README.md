# icra2027_underice — internal research log

**External readers: go to [`release/`](release/README.md).** That folder is
self-contained: a 2-page PDF explaining the goal and result, plus the three
short evaluation scripts that reproduce every number. You do not need anything
else in this directory.

---

## What this folder is

A full research log for the ICRA 2027 under-ice 3D reconstruction project,
kept in roughly chronological order. It contains intermediate analyses,
dead ends, re-scopes, and superseded findings alongside the current ones.
It is not organized for external consumption — use `release/` for that.

## Folder map

| Folder | Contents |
|--------|----------|
| `release/` | Clean, external-facing package. Start here. |
| `results/` | Per-investigation markdown write-ups and figures. See legend below. |
| `scripts/` | All analysis scripts, including internal/diagnostic ones. |
| `configs/` | Station configs (internal paths). |

## Legend for `results/` file naming

- **`TASK<N>_*`** — a broader investigation thread. Task 0 was the original
  multibeam-comparison plan; Task 1 was the re-scoped sonar-accuracy test.
- **`ITEM<N>_*`** — a specific numbered claim being quantified (e.g., sonar
  coverage ablation, per-region breakdown, runtime costs).
- **`PROOF<N>_*`** — a proof-of-concept test for a single specific claim
  (referee result, FRC coherence).
- **`*_DIAGNOSTIC.md`** — exploratory, usually not cited in the paper.

## What is superseded / abandoned

| File | Status |
|------|--------|
| `TASK0_VERDICT.md` | Superseded by `TASK0_TASK1_RESCOPE_DECISION.md`. The original Task 0 compared optical vs. multibeam sonar surfaces — abandoned when it was confirmed three independent ways that no multibeam instrument (Imagenex DT101 or any other) exists in this dataset. Only a Valeport single-beam altimeter is present. |
| `TASK7_DATASET_MANIFEST.md` | Documents the dataset re-scoping; explains why results referencing "DT101" or "480 beams" are no longer valid. |
| `SONAR_VERIFICATION_PS117_29.md` | PS117-29 (harder station). MASt3R init was right-signed (r=+0.067) but unconstrained photometric training corrupted it to r=−0.31. Included as evidence that vision-only corruption is repeatable across stations, not just PS117-39. Not in the main result table — different station. |

## Current headline finding

Sonar-supervised 3DGS vs. two vision-only baselines on PS117-39:

| Method | Sonar r (held-out) | Sonar RMSE |
|--------|--------------------|------------|
| Ours (sonar-supervised) | +0.973 | 0.181 m |
| SeaSplat (no sonar) | −0.861 | 1.340 m |
| WaterSplatting (no sonar) | −0.859 | 1.274 m |

Both baselines are sign-inverted: as real range increases, their rendered depth
decreases. This is a structural property of vision-only optimization on
low-texture ice, observed independently across two different codebases.
Full methodology in `release/Sonar_Story.pdf`.

## What is NOT in this folder

- Raw video or image sequences (stored externally on F:/)
- Sonar data files (`valeport.dat`, stored externally on F:/)
- Trained 3DGS checkpoints (stored externally, paths in `configs/`)
- WaterSplatting / nerfstudio training output (stored on D:/)
- Training code (see the respective upstream repos: gaussian-splatting,
  SeaSplat, WaterSplatting)
