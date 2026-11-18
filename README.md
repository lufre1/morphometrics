# mito-morphometrics

Per-mitochondrion and per-cell (axon) morphometrics from EM segmentations, in 2D and 3D,
plus the statistics and figure scripts built on top of them (WT vs Plp-deficient optic nerve,
manual IMOD tracing vs automatic CNN segmentation).

## Environment

`env.yaml` only covers `morphometrics_2d.py`:

```bash
conda env create -f env.yaml
conda activate mito-morphometrics
```

Everything else (3D pipeline, figures, stats) runs in the `synapse` env, which additionally
provides `zarr`, `scipy`, `matplotlib`, `statsmodels`, `Pillow`, `napari` and `synapse.util`:

```bash
mamba activate synapse
```

## Morphometrics

| Script | What it does |
|---|---|
| `morphometrics_2d.py` | 2D: area, perimeter, circularity, axes, intensity per mito; optional per-cell summary. |
| `morphometrics_3d.py` | 3D: volume, surface, sphericity, S/V, elongation/flatness, nearest neighbour, … per mito; per-cell summary and QC. TIFF or Zarr v2 input. See the module docstring for the 2D↔3D metric mapping. |

```bash
# 3D, Zarr input, restricted to a z-window
python morphometrics_3d.py \
    -p raw.zarr --raw_key s0 \
    -mlpth mito.zarr --mito_key seg \
    -clpth axons.zarr --cell_key s2 \
    --voxel_size 25 5 5 -zs 252 -ze 1252 \
    -o results/
```

`python morphometrics_3d.py -h` lists all options. Notes:

- Voxel size is given in nm (z y x); outputs are in µm.
- `--path` is optional: label-only runs drop the intensity columns and the intensity QC gate.
- `-zs/-ze` restrict the *whole* analysis: mitos crossing a window face are dropped, and the
  cell volume counts only in-window voxels (see `mito_morphometrics_zwindow_audit.md`).
- Axon segmentations are native at `s2`; use `--cell_key s2`.

Outputs in `<output_path>/`:

- `mito_morphometrics_3d.csv` — one row per mitochondrion
- `mito_morphometrics_3d_qc.csv` — after QC filtering
- `cell_summary_3d.csv` — one row per cell (with `-clpth`)
- `cell_level_mito_summary.csv` — mito statistics aggregated per cell

### Batch runs

Paths are hard-coded to `~/data/volume-em`. Results are mirrored into `data/`, where the figure
scripts read from.

| Script | Arm | Output |
|---|---|---|
| `run_morphometrics_imod.sh` | manual (IMOD), 6 animals, full extent + 25 µm window | `data/<id>_morphometrics_imod_{full,z25}/` |
| `run_morphometrics_auto.sh` | automatic (CNN), 4007 + 4010, 25 µm window | `data/<id>_morphometrics_auto_z25/` |

```bash
VARIANTS=z25 ./run_morphometrics_imod.sh   # subset of variants
ONLY=4010 ./run_morphometrics_auto.sh      # single animal
```

### Checks

```bash
python test_zrange.py                 # z-window arithmetic in morphometrics_3d
python verify_morphometrics_imod.py   # sanity checks on the IMOD runs
python figures_3d.py --selfcheck      # published p-values still reproduce
python figure_pipeline_forest.py --selfcheck
```

## Figures and statistics

All read from `data/` and write PDFs to `Plots/` (or `figures/`) and tables to `table/`.
Add `--qc` to use the QC-filtered mito tables, `--png` for a PNG copy; `-h` for the rest.

| Script | Figure |
|---|---|
| `figures_3d.py` | Fig. 1 (axon level) and Fig. 2 (mito level), WT vs KO |
| `figures_screenshots.py` | 3D reconstruction screenshots, 4007 vs 4010 |
| `figure_combined.py` | reconstructions + mito-level panels in one figure |
| `figure_manual_vs_auto.py` | manual vs automatic segmentation, by genotype |
| `figure_pipeline_forest.py` | genotype effect size per pipeline (forest plot) |
| `aggregate_overview.py` | cross-animal overview CSVs (no figure) |
| `inspect_overlap_napari.py` | napari viewer: where auto segmentation overshoots manual (4010) |

Per-metric statistics scripts (Mann-Whitney U, GEE; plots + tables) live in
`<region>/<condition>/<metric>` — `RGC/`, `LGN/`, `N.opt./` (2D) and `N.opt._3D/` (3D).
They are extensionless Python files: run them with `python "N.opt._3D/Plp-deficient/volume"`.

`data/`, `figures/`, `table/`, `Plots/` and `*.csv` are git-ignored.

## License

See `LICENSE`.
