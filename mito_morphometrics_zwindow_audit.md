# Audit: the z-window was applied to only half of the automatic analysis

**Date:** 2026-09-16
**Scope:** the automatic (CNN) morphometrics arm, animals 4007 (KO) and 4010 (WT).
**Outcome:** two defects found, both confirmed against the raw segmentations; the arm was
re-run and every downstream figure and table regenerated.

Companion document: `~/data/volume-em/mito_morphometrics_findings.md` holds the biological
findings. Its manual-vs-automatic section and caveat 2 were rebuilt from the regenerated CSVs on
the same day; see [Action list](#action-list).

---

## The two defects

The published `<id>_morphometrics_final/` tables were produced in 2026-05 by an older script
(`scripts/volume-em_analysis/morphometrics_3d_claude.py`). Both animals nominally cover a 25 µm
z-window — 4007 slices 347–1347, 4010 slices 252–1252.

### 1. The density denominator was never restricted to the window

`-zs/-ze` filtered the **mitochondria list only**. `cell_volume_um3`, the denominator of every
density and volume fraction, is the axon volume over the **full traced extent**.

Confirmed two independent ways:

1. `cell_volume_um3` in `_morphometrics_final` is bit-identical to the volume in the
   unrestricted `_morphometrics_all_final` run, for all 865 (4007) and 885 (4010) axons.
2. In-window axon volumes recomputed directly from the segmentations give a consistent
   overstatement:

| animal | reported (full extent) ÷ true (in-window) |
|---|---|
| 4007 | 1.13× – 1.54×, mean 1.42× |
| 4010 | 1.27× – 1.58×, mean 1.42× |

Every automatic density, volume fraction and membrane-area density was therefore **25–35 % too
low**, by an amount that varies per axon.

### 2. The two pipelines used different z-inclusion rules

- automatic (`_final`, pre-fix script): a mitochondrion is kept if its bounding box **overlaps**
  the window.
- manual (`_imod_z25`, current script): kept only if it lies **entirely inside** the window.

Evidence: in 4007 the mito centroids span 300.7–1372.7 for a window of 347–1347 (14 objects have
their centroid outside the window entirely); 4010 spans 128.7–1283.2 for 252–1252 (15 outside).
Scanning the 4010 mito segmentation outside the window: **28 of 242 objects (11.6 %) protrude**,
carrying **21.5 % of the total mitochondrial volume** — long objects straddle most often.

**The QC table does not remove them.** `touches_border` means the edge of the *volume*, not the
faces of the window; the QC filter never knew a window existed. Of the 28 protruding objects in
4010, QC removes 4 (for `elongation ≥ 20`) and keeps 24 — still 10.2 % of the QC count and
18.2 % of the QC mitochondrial volume. 4007's `_qc` table is identical to its raw table: 127 rows
in, 127 rows out, nothing dropped at all.

Consequently the raw mitochondria count could not be compared between pipelines even at the
matching 25 µm window, because the two arms were counting under different rules.

---

## The fix

`run_morphometrics_auto.sh` (new) re-runs both animals through the current
`morphometrics_3d.py`, which since 2026-09 restricts the whole analysis: objects crossing a
window face are dropped rather than truncated, and the axon volume counts only in-window voxels.
Output goes to `<id>_morphometrics_auto_z25`, alongside the old tables rather than over them.

Published z-windows and 10-axon whitelists were kept, so the new tables remain comparable to the
existing per-run READMEs.

`--cell_key s2` for 4007 is deliberate: **s2 is the native resolution of the axon segmentation**
and s0 is only an upsampled copy. Axon volumes recomputed from s2 reproduce the published
`cell_volume_um3` exactly, which also confirms s2 is what the original run used.

---

## Verification of the re-run

Two exact checks:

- **Denominator.** All 20 new `cell_volume_um3` values match the in-window axon volumes measured
  independently from the segmentations, to every printed digit (max deviation 0.0000 %).
- **Numerator.** The new mito set is a strict subset of the old one — 214 of 242 (4010), 103 of
  127 (4007) — and every surviving object has an **identical volume**, maximum difference exactly
  0. The 28 objects dropped in 4010 are precisely the protruding objects identified independently
  by scanning outside the window.

So the only things that changed are which objects are included and what they are divided by.

`figures_3d.py --selfcheck`, `figure_pipeline_forest.py --selfcheck` and `test_zrange.py` all
pass after the constant refresh described below.

---

## What changed in the numbers

### Automatic arm totals

| | 4007 (KO) | 4010 (WT) |
|---|---|---|
| mitochondria | 127 → 103 (−19 %) | 242 → 214 (−12 %) |
| mitochondrial volume | 17.71 → 14.13 µm³ (−20 %) | 19.65 → 15.42 µm³ (−22 %) |
| axon volume | 264.5 → 189.0 µm³ (−29 %) | 390.6 → 274.4 µm³ (−30 %) |
| pooled volume fraction | 0.0670 → **0.0747** | 0.0503 → **0.0562** |

### Figures 1 and 2 (WT vs KO), raw mode

| panel | metric | p before → after |
|---|---|---|
| F1A | median mito elongation | 0.734 → 0.910 |
| F1B | median mito flatness | 0.076 → 0.089 |
| F1C | mito volume density | 0.064 → 0.064 |
| F1D | mito count density | 0.345 → 0.140 |
| F2A | mito volume | 0.121 → 0.054 |
| F2B | sphericity | 0.345 → 0.273 |
| F2C | surface-to-volume ratio | 0.002 → 0.001 |
| F2D | nearest-neighbour distance | 0.678 → 0.385 |
| F2E | elongation (pooled) | 0.217 → 0.551 |
| F2F | flatness (pooled) | 0.074 → 0.093 |

Two panels cross 0.05, both in **QC mode**: F2A mitochondrial volume 0.089 → **0.026** (becomes
significant) and F2F flatness 0.035 → **0.068** (stops being significant).

F2A's WT value is a median of 10 per-axon medians and moved a long way (0.0649 → 0.0357) for a
12 % change in object count. That statistic is fragile at n = 10; it is not evidence of a further
problem.

### Manual vs automatic, matched animals

| panel | comparison | before → after |
|---|---|---|
| C | WT volume density, auto | 5.15 → 5.72 per 100 µm³ (manual 3.36); p 0.0047 → 0.0007 |
| C | KO volume density, auto | 6.49 → 7.65 per 100 µm³ (manual 7.02); p 0.594 → 0.859 |
| D | WT count density, auto | 56.3 → 66.7 per 100 µm³ (manual 37.1) |
| D | KO count density, auto | 42.8 → 50.8 per 100 µm³ (manual 91.7) |
| E | genotype effect on raw count, within auto | p 0.045 → 0.031 |

The WT gap between pipelines widens; the KO gap narrows.

### Pipeline forest, `auto(2)` arm

The two IMOD arms are unchanged, as expected — they never touched automatic data.

| metric | ratio before → after | p before → after |
|---|---|---|
| mito count density | 0.761 → 0.762 | 0.345 → 0.140 |
| mito volume fraction | 1.261 → 1.337 | 0.064 → 0.064 |
| membrane area density | 0.966 → 1.054 | 0.970 → 0.734 |
| median mito volume | 1.825 → 3.146 | 0.121 → 0.054 |
| median S/V ratio | 0.820 → 0.753 | 0.002 → 0.001 |

### Matched-mitochondria comparison

Because the automatic object set shrank, the shared-axon sets changed: 4010 went from axons
{1, 5} to {1}. Current values:

| animal | shared axons | manual mitos there | matched within 0.5 µm | median auto/manual volume |
|---|---|---|---|---|
| 4007 | 1, 2, 3 | 55 | 71 % | 1.57× |
| 4010 | 1 | 36 | 44 % | 1.34× |

The headline conclusion is unchanged in direction and magnitude: **the automatic segmentation
overestimates mitochondrial volume by roughly 35–55 % relative to manual tracing.**

---

## Code changes made

| file | change |
|---|---|
| `run_morphometrics_auto.sh` | new; rebuilds the automatic arm for both animals |
| `aggregate_overview.py` | reads each variant's *own* `cell_summary_3d.csv` (it used the full run's volumes for the z25 variant too, while `z25_fractions_comparable` had already been flipped to `True`); adds `axon_volume_um3_z25`; auto arm → `_auto_z25` |
| `figures_3d.py` | `SAMPLES` → `_auto_z25`; `EXPECTED_P` refreshed, superseded values kept in a comment |
| `figure_manual_vs_auto.py` | `sample_dirs` → `_auto_z25`; docstring now records what the old `_final` tables actually did |
| `figure_pipeline_forest.py` | `Z_CAVEAT` rewritten — it still carried the pre-fix wording while `LEGEND_TEXT` carried the post-fix wording, and both printed into the same output; dead "confound 1" reference removed; auto arm → `_auto_z25`; `auto(2)` selfcheck constant refreshed |

---

## Action list

### Delete or archive — these are wrong, not merely old

```
data/4007_morphometrics_final/      ~/data/volume-em/KO/4007/4007_morphometrics_final/
data/4010_morphometrics_final/      ~/data/volume-em/WT/4010/4010_morphometrics_final/
```

`cell_level_mito_summary.csv` and `cell_summary_3d.csv` divide a windowed mitochondria count by a
full-extent axon volume; the mito tables contain objects that merely overlap the window. Their
`README.md` states a z-range as though the whole analysis were restricted, which it was not. No
script in this repository reads them any more.

### Updated on 2026-09-16 (done)

- `~/data/volume-em/mito_morphometrics_findings.md` — the "Manual (IMOD) vs automatic
  segmentation" section was rebuilt from the regenerated CSVs (it quoted the biased automatic
  numbers 48.0 and 62.0 per 100 µm³ and match rates of 85 % / 59 %). Caveat 2 was rewritten: the
  `z25` densities are usable now and the flag reads `True`. Caveat 9 now points at `_auto_z25` and
  quotes 214 rather than 242 automatic mitochondria.
- `~/data/volume-em/WT/4010/4010_morphometrics_all_final/README.md` — the opening `## Command`
  block held a *visualisation* command containing `-zs 252 -ze 1252 -fid ...`, which made an
  unwindowed, whole-volume table look windowed and whitelisted. It is now labelled as a rendering
  command and preceded by an explicit scope section. The `_all_final` tables themselves are fine:
  no window was applied, so numerator and denominator are both full-volume and self-consistent.

### Outside this repository

Any figure or number already exported into a manuscript or slides. In particular the published
figure-1 p-values **0.850 / 0.054 no longer reproduce in any mode** — the QC column used to match
the published figure only because that figure was built on the biased tables.

### Loose end worth one check

`4007_imod.zarr` is z = 1448; the automatic 4007 volume is z = 1447. `aggregate_overview.py` looks
up automatic mitochondria centroids inside the manual axon labels, so it assumes the two share a z
origin. One slice in 1447 is small, but confirm which end the extra slice sits on before quoting
the matched-mitochondria numbers.

---

## Reproducing this audit

```bash
mamba activate synapse
./run_morphometrics_auto.sh                 # ONLY=4007 to run one animal
python aggregate_overview.py
python figures_3d.py            ; python figures_3d.py --qc
python figure_manual_vs_auto.py ; python figure_manual_vs_auto.py --qc
python figure_pipeline_forest.py --imod-variant full
python figure_pipeline_forest.py --imod-variant z25
python figures_3d.py --selfcheck ; python figure_pipeline_forest.py --selfcheck
python test_zrange.py
```
