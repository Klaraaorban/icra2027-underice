# Task 7 — Dataset Labelling Audit

**Status: COMPLETE.** Traced from raw source files, not from folder names or prior
session assumptions. Time budget: ~30 min as specified.

## Finding 1 (confirms the brief): hemisphere and expedition

All evidence internally consistent and points to **RV Polarstern expedition PS117,
Weddell Sea sector, Antarctic, January 2019** — NOT MOSAiC/PS122 (Arctic). This is
a **different hemisphere and a different expedition** from the earlier workshop
paper's PS122/3 39-20 data.

| Station | Lat (deg) | Lon (deg) | Hemisphere | Region |
|---|---|---|---|---|
| PS117-29 | -69.10 | -0.04  | S | Weddell Sea, Atlantic Antarctic sector (near Prime Meridian) |
| PS117-37 | -70.20 | -11.25 | S | Weddell Sea, Atlantic Antarctic sector |
| PS117-39 | -70.30 | -10.04 | S | Weddell Sea, Atlantic Antarctic sector |

Source: `PS117-{station}-Summary.csv`, columns `Latitude (degrees)` /
`Longitude (degrees)`, first valid row per station.

Corroborating text evidence: `PS117-29/README.txt` — "This was the first ROV
deployment on Polarstern... Mark Milnes - 5 January 2019" — a first-person,
dated field note consistent with a January 2019 Polarstern cruise, not a later
MOSAiC (2019/20, different vessel) deployment.

**Paper framing implication**: since Klara's earlier workshop paper used MOSAiC
(Arctic, PS122), and this project uses PS117 (Antarctic, Weddell Sea), a
cross-hemisphere generalization claim ("the method transfers between Arctic and
Antarctic under-ice conditions") is available and should be stated explicitly —
per the brief, this is a strength, not a footnote.

## Finding 2 (NEW, not in the brief): folder-name dates are wrong and must not be used

Every station's folder name embeds a date that does **not** match the actual
survey/logging timestamp. The true timestamp was recovered two independent ways
per station — the datalogger export XML's Unix `<starttime>`, and the first row
of the summary CSV — and they agree with each other but disagree with the folder
name by 1-3 days in every case:

| Station | Folder name says | Actual survey date (XML + CSV, agree) | Discrepancy |
|---|---|---|---|
| PS117-29 | `..._2019-01-06_15-22-55` | **2019-01-03**, ~19:42-? UTC | folder date is 3 days late |
| PS117-37 | `..._2019-01-10_11-11-06` | **2019-01-09**, 18:24-? UTC | folder date is 1 day late |
| PS117-39 | `..._2019-01-12_14-15-45` | **2019-01-10**, 12:18-13:33 UTC | folder date is 2 days late |

Interpretation: the folder-name timestamp is almost certainly when the datalog
was **exported/downloaded** from the logging system (`order_<id>.xml`'s creation
context), not when the ROV dive happened. This is a real, previously-undocumented
data hygiene issue.

**Practical impact — low, but must be stated explicitly in the paper's dataset
section**: every sonar-sync constant (`CLIP_START` / `CLIP3_START`) used
throughout this project's pipeline was derived empirically by matching against
the CSV/valeport timestamps directly, not against folder names, so **no
pipeline code needs to change**. But any future person reading a folder name and
assuming it's the deployment date will be wrong by 1-3 days, and any figure
caption or supplementary material that states a per-station date should cite the
CSV/XML date, never the folder name.

## Finding 3: sequence identity check

Confirmed the 300-frame windows actually used in this project's pipeline come
from the source video paths their configs claim:
- PS117-39 easy tier: `PS117-39/gopros/.../3D_L0001.MP4` region t=30-90s, source
  video's own recording is within the 2019-01-10 window above.
- PS117-29 hard tier: `PS117-29/gopros/PORT/DCIM/100S3D_L/3D_L0001.MP4` region
  t=930-990s, source video's own recording is within the 2019-01-03 window above.

No mismatch found between the video files actually loaded by the training
pipeline and the station folders they're labelled as belonging to.

## Finding 4 (2026-08-30, later): the ranging sensor is confirmed, and it is NOT a multibeam

Investigated per Klara's request to verify rather than assume. The PS117 cruise
report's ROV-sampling section (project SIPES 2, co-authored by M. Milnes — the
same person named in this dataset's own `README.txt`) describes this specific
ROV's payload as hyperspectral radiometers + video + stills camera only,
exactly matching this dataset's file inventory. This is a different, simpler
ROV deployment than the Imagenex DT101-equipped platform in Katlein et al. 2017
(Frontiers Mar. Sci. 4:281), which that paper's own text ties specifically to
the later 2019/20 MOSAiC expedition. The Valeport instrument is identified with
high confidence as a **Valeport VA500P Altimeter** (integrated acoustic +
pressure sensor — its NMEA sentence's paired Ping-Depth/Pressure-Depth fields
match the VA500P's documented option exactly): 500kHz, beam angle ±3°, native
ping rate confirmed from the raw file's own timestamps at 3.92Hz (matches the
datasheet's "4Hz" option). Full reasoning and sources in
[TASK0_VERDICT.md](TASK0_VERDICT.md), which this finding blocked/unblocked.

## Action items

1. Use "RV Polarstern PS117 (Weddell Sea, Antarctic, Jan 2019)" as the
   expedition/hemisphere framing throughout the paper — confirmed, not assumed.
2. Add a one-line caveat to the dataset section: "station folder names in the raw
   AWI export carry the datalog export date, not the survey date; all dates
   reported here are recovered from datalogger timestamps."
3. State the Arctic (MOSAiC/PS122, prior workshop paper) vs. Antarctic (PS117,
   this paper) distinction explicitly as a generalization strength.
