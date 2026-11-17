#!/usr/bin/env python
"""Manual (IMOD) vs automatic segmentation, side by side with the genotype comparison.

Four violins per panel, grouped by genotype:

        WT (4010)              Plp-/y (4007)
    manual   automatic      manual   automatic
    (pale)    (full)        (pale)    (full)

Colour encodes genotype, exactly as in figures_3d.py, so this figure sits next to the
others without re-teaching the reader the palette. Saturation encodes the pipeline:
the hand-traced data are drawn in a paler, greyer version of the same hue.

Both comparisons are therefore readable in one panel -- genotype left vs right, pipeline
within each pair -- and the genotype bracket is drawn once per pipeline, which shows
directly whether the genotype effect replicates across segmentation methods.

Only 4007 (KO) and 4010 (WT) were processed by both pipelines. --manual-animals decides
what the manual violins hold:

Panel E of the axon-level figure is the raw mitochondria count per axon, the one number
that says directly whether the automatic pipeline finds as many objects as the hand
tracing. Being an absolute count it is the one panel that cannot survive unequal z
coverage: it is honest only at the default --imod-variant z25, which matches the
automatic 25 um z-window. With --imod-variant full the manual side covers 26-39 um and
its count is inflated by z coverage, not by the pipeline. Panel D is the volume-corrected
read of the same quantity and is fine under either variant.

The other panels are variant-independent for two different reasons. A and B are medians
of per-object shape, which do not scale with how much tissue you sample (measured 1.00x
and 0.99x between the two variants). C and D are ratios whose numerator and denominator
are measured over the same z-window in BOTH arms -- true only since the 2026-09 fix to
morphometrics_3d.py, which made -zs/-ze crop the axon volume as well as the mito list.
The manual side got that fix by re-running run_morphometrics_imod.sh; the automatic side
needed run_morphometrics_auto.sh, because the published *_morphometrics_final tables were
built by the pre-fix script and divide a windowed mito count by a full-extent axon volume
(low by 1.13-1.58x) while keeping every mito that merely OVERLAPS the window (in 4010,
28 of 242 objects carrying 21.5% of the mito volume). Those tables are superseded by
*_morphometrics_auto_z25 and must not be mixed with z25 manual data.

  matched (default)  those same two animals, so manual and automatic describe the same
                     tissue and a difference between them is the pipeline
  all                the whole IMOD dataset, 3 animals per genotype. The automatic side
                     still has 1 animal each, so the two sides are NOT matched samples
                     and a manual-vs-automatic difference there mixes pipeline with animal

Either way the axons are pseudo-replicates -- one animal per genotype on the automatic
side -- so every p-value is labelled exploratory, the convention figures_3d.py uses.

Usage
-----
    python figure_manual_vs_auto.py                       # imod z25 vs automatic, raw tables
    python figure_manual_vs_auto.py --qc --png
    python figure_manual_vs_auto.py --imod-variant full   # full traced extent instead
    python figure_manual_vs_auto.py --manual-animals all  # all six hand-traced animals
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from figures_3d import (KO_COLOR, WT_COLOR, add_bracket, build_axon_table,  # noqa: E402
                        load_cell_volumes, load_mito_level, mwu)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

GENOTYPES = ("WT", "KO")
PIPELINES = ("manual", "auto")
GENOTYPE_LABELS = {"WT": "WT", "KO": r"$\mathit{Plp}^{-/y}$"}
PIPELINE_LABELS = {"manual": "manual", "auto": "automatic"}

# Automatic morphometrics exist for these two animals only; 4005, 4009, 4015 and 4016
# were hand-traced but never run through the automatic pipeline.
AUTO_ANIMALS = {"WT": ["4010"], "KO": ["4007"]}

# Which hand-traced animals go into the manual violins.
#   matched -- only the two animals the automatic pipeline also covers, so manual and
#              automatic describe the same tissue and a difference is the pipeline.
#   all     -- the whole IMOD dataset, three animals per genotype. The automatic side
#              still has one animal each, so the two sides are NOT matched samples.
MANUAL_SETS = {
    "matched": {"WT": ["4010"], "KO": ["4007"]},
    "all": {"WT": ["4009", "4010", "4016"], "KO": ["4005", "4007", "4015"]},
}

# x positions: genotypes separated by a gap, pipelines paired inside each genotype
POS = {("WT", "manual"): 0.0, ("WT", "auto"): 1.05,
       ("KO", "manual"): 2.15, ("KO", "auto"): 3.2}
GROUP_CENTRE = {"WT": 0.525, "KO": 2.675}

BASE_COLOR = {"WT": WT_COLOR, "KO": KO_COLOR}


# How far the manual (hand-traced) colours are pulled toward light grey. Tuned so the
# two pipelines are obviously different at a glance: the genotype hue stays readable,
# but the manual violins, boxes and dots are clearly the lighter, greyer pair.
MANUAL_MIX = 0.68     # 0 = identical to the automatic colour, 1 = pure grey
MANUAL_GREY = 0.86    # the grey blended toward (0 = black, 1 = white)


def _pale(color, mix=MANUAL_MIX, grey=MANUAL_GREY):
    """A much lighter, greyer version of a hue: blend it toward light neutral grey.

    Keeps the genotype hue recognisable while marking the hand-traced data as the
    quieter of the two pipelines.
    """
    return tuple((1 - mix) * c + mix * grey for c in mcolors.to_rgb(color))


COLOR = {(g, "auto"): BASE_COLOR[g] for g in GENOTYPES}
COLOR.update({(g, "manual"): _pale(BASE_COLOR[g]) for g in GENOTYPES})


# Per-mitochondrion panels, mirroring figure 2 of figures_3d.py.
# Elongation and flatness are tested pooled over all mitochondria for the same reason
# given there: shape varies too much within one axon for a median to mean anything.
MITO_PANELS = [
    dict(panel="A", letter="A", title="3D mitochondrial volume",
         ylabel="Mitochondrial volume (µm³)", column="volume_um3"),
    dict(panel="B", letter="B", title="3D mitochondrial sphericity",
         ylabel="Mitochondrial sphericity", column="sphericity"),
    dict(panel="C", letter="C", title="3D surface-to-volume ratio",
         ylabel="Surface-to-volume ratio (µm⁻¹)", column="sv_ratio"),
    dict(panel="D", letter="D", title="3D nearest-neighbour distance",
         ylabel="Nearest-neighbour distance (µm)", column="nearest_neighbor_um"),
    dict(panel="E", letter="E", title="3D mitochondrial elongation",
         ylabel="Mitochondrial elongation (a/c)", column="elongation_a_over_c",
         test_level="pooled", p_suffix=" (pooled)"),
    dict(panel="F", letter="F", title="3D mitochondrial flatness",
         ylabel="Mitochondrial flatness (b/c)", column="flatness_b_over_c",
         test_level="pooled", p_suffix=" (pooled)"),
]

# Per-axon panels, mirroring figure 1 of figures_3d.py.
AXON_PANELS = [
    dict(panel="A", letter="A", title="3D median mitochondrial elongation per axon",
         ylabel="Median mitochondrial elongation (a/c)", column="median_mito_elongation"),
    dict(panel="B", letter="B", title="3D median mitochondrial flatness per axon",
         ylabel="Median mitochondrial flatness (b/c)", column="median_mito_flatness"),
    dict(panel="C", letter="C", title="3D axonal mitochondrial volume density",
         ylabel="Mitochondrial volume density (µm³ / 100 µm³ axon)",
         column="mito_volume_density_per100"),
    dict(panel="D", letter="D", title="3D axonal mitochondrial count density",
         ylabel="Mitochondrial density (mitochondria / 100 µm³)",
         column="mito_count_density_per100"),
    # Raw count, uncorrected for axon volume -- only honest at --imod-variant z25, see
    # the module docstring. Panel D is the volume-corrected read of the same thing.
    dict(panel="E", letter="E", title="3D mitochondria per axon (z25 only)",
         ylabel="Mitochondria per axon (count)", column="mito_count"),
]

def sample_dirs(pipeline, variant, manual_animals):
    """-> {result directory: genotype} for one pipeline."""
    if pipeline == "auto":
        return {f"{a}_morphometrics_auto_z25": g
                for g, animals in AUTO_ANIMALS.items() for a in animals}
    return {f"{a}_morphometrics_imod_{variant}": g
            for g, animals in MANUAL_SETS[manual_animals].items() for a in animals}


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def load_both(data_dir, qc, imod_variant, manual_animals):
    """-> (mito_df, axon_df, source_tables) with a `pipeline` column on each."""
    mito_frames, axon_frames, tables = [], [], {}
    for pipeline in PIPELINES:
        samples = sample_dirs(pipeline, imod_variant, manual_animals)
        mito, table = load_mito_level(data_dir, qc, samples)
        cells = load_cell_volumes(data_dir, samples)
        # cell_id repeats across pipelines, so keep the axon key pipeline-unique
        mito["axon_uid"] = pipeline + "__" + mito["axon_uid"]
        cells["axon_uid"] = pipeline + "__" + cells["axon_uid"]
        cells["pipeline"] = pipeline
        mito["pipeline"] = pipeline
        axon = build_axon_table(cells, mito)
        mito_frames.append(mito)
        axon_frames.append(axon)
        tables[pipeline] = f"{os.path.basename(list(samples)[0]).split('_', 1)[1]}/{table}"
    return (pd.concat(mito_frames, ignore_index=True),
            pd.concat(axon_frames, ignore_index=True), tables)


def _values(df, column, genotype, pipeline):
    sel = (df.genotype == genotype) & (df.pipeline == pipeline)
    return df.loc[sel, column].dropna().to_numpy()


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

# Two stacked brackets sit at these fractions of the axis height; the axis top is set
# from the data so the brackets clear it without leaving a band of empty plot.
BRACKET_FRAC = (0.66, 0.84)
HEADROOM = 1.6


def _fmt_p(p):
    """Never print 'p = 0.000'."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _nice_top(data_max):
    """Axis top: just enough room above the data for two brackets and their labels."""
    return float(data_max * HEADROOM) if data_max > 0 else 1.0


def group_label(genotype, manual_animals):
    """Genotype label under each violin pair.

    In `matched` mode both pipelines describe the same animal, so naming it is honest.
    In `all` mode they do not, so the animal composition moves to the figure legend
    rather than implying the pair is one animal.
    """
    if manual_animals == "matched":
        return f"{GENOTYPE_LABELS[genotype]} ({MANUAL_SETS['matched'][genotype][0]})"
    return GENOTYPE_LABELS[genotype]


def panel4(ax, spec, axon_vals, mito_vals, test_vals, animal_vals, manual_animals,
           jitter_seed=0):
    """Draw one four-violin panel.

    Returns ({(comparison): (u, p)}, {pipeline: (p_per_animal, n_wt, n_ko)}).

    Each bracket shows the test the panel draws, over axons or over pooled objects; the
    unit is left to the caption. The same contrast with each animal reduced to one value
    -- the only independent unit -- is computed and returned for the stats table but not
    drawn.
    """
    keys = [(g, p) for g in GENOTYPES for p in PIPELINES]
    dist = [mito_vals[k] if mito_vals is not None else axon_vals[k] for k in keys]
    positions = [POS[k] for k in keys]
    colors = [COLOR[k] for k in keys]

    parts = ax.violinplot(dist, positions=positions, showmeans=False, showmedians=False,
                          showextrema=False, widths=0.6, bw_method=0.3, points=200)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.0)

    box = ax.boxplot(dist, positions=positions, widths=0.18, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    for line in box["whiskers"] + box["caps"]:
        line.set_linewidth(1.2)

    rng = np.random.default_rng(jitter_seed)
    if mito_vals is not None:
        for k in keys:                      # small points: individual mitochondria
            v = mito_vals[k]
            ax.scatter(rng.normal(POS[k], 0.08, len(v)), v, s=10, alpha=0.35,
                       color=COLOR[k], zorder=2)
    for k in keys:                          # large white-ringed points: one per axon
        v = axon_vals[k]
        ax.scatter(rng.normal(POS[k], 0.05, len(v)), v, s=90, color=COLOR[k],
                   edgecolor="white", linewidth=0.8, zorder=4)

    top = _nice_top(max(np.max(v) for v in dist))
    stats, animal_stats = {}, {}
    # one genotype bracket per pipeline: does the WT/KO effect replicate?
    for pipeline, frac in zip(PIPELINES, BRACKET_FRAC):
        u, p = mwu(test_vals[("WT", pipeline)], test_vals[("KO", pipeline)])
        stats[("genotype", pipeline)] = (u, p)
        n_wt = len(animal_vals[("WT", pipeline)])
        n_ko = len(animal_vals[("KO", pipeline)])
        # one animal per genotype: genotype and individual are perfectly confounded,
        # so there is no per-animal test to report.
        p_animal = (mwu(animal_vals[("WT", pipeline)], animal_vals[("KO", pipeline)])[1]
                    if min(n_wt, n_ko) >= 2 else np.nan)
        animal_stats[pipeline] = (p_animal, n_wt, n_ko)
        add_bracket(ax, POS[("WT", pipeline)], POS[("KO", pipeline)], frac * top, 0.018 * top,
                    _fmt_p(p), fontsize=15)
    # pipeline comparisons are recorded but not drawn, to keep the panel readable
    for g in GENOTYPES:
        stats[("pipeline", g)] = mwu(test_vals[(g, "manual")], test_vals[(g, "auto")])

    ax.set_title(spec["title"], fontsize=16, pad=12)
    ax.set_ylabel(spec["ylabel"], fontsize=15)
    ax.set_xticks([POS[k] for k in keys])
    ax.set_xticklabels([PIPELINE_LABELS[p] for _g, p in keys], fontsize=14)
    ax.set_xlim(-0.45, 3.65)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    for g in GENOTYPES:                     # genotype label under each pair
        ax.text(GROUP_CENTRE[g], -0.16, group_label(g, manual_animals),
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=5, labelsize=14)
    ax.text(-0.18, 1.06, spec["letter"], transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom", ha="left")
    return stats, animal_stats


def legend_handles(manual_animals, with_mito=True):
    """Figure-level legend: dot tiers plus the pale/full pipeline swatches."""
    if manual_animals == "matched":
        axon_label = "one axon (manual n = 5, automatic n = 10 per animal)"
    else:
        # the two sides are not matched samples here, so say exactly what feeds each
        wt, ko = (", ".join(MANUAL_SETS["all"][g]) for g in ("WT", "KO"))
        axon_label = ("one axon — manual: 3 animals x 5 axons "
                      f"(WT {wt}; $\\mathit{{Plp}}^{{-/y}}$ {ko});\n"
                      "automatic: 1 animal x 10 axons (WT 4010; "
                      "$\\mathit{Plp}^{-/y}$ 4007)")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="0.35", markeredgecolor="none", alpha=0.55,
               label="one mitochondrion"),
        Line2D([], [], marker="o", linestyle="none", markersize=9.5,
               markerfacecolor="0.35", markeredgecolor="white", markeredgewidth=0.8,
               label=axon_label),
        Patch(facecolor=_pale("0.45"), edgecolor="black", alpha=0.6, label="manual"),
        Patch(facecolor="0.45", edgecolor="black", alpha=0.6, label="automatic"),
    ]
    return handles if with_mito else handles[1:]


def build_figure(panels, rows_out, figure_name, axon_df, mito_df, tables,
                 nrows, ncols, figsize, manual_animals):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    n_animals = {("manual", g): len(MANUAL_SETS[manual_animals][g]) for g in GENOTYPES}
    n_animals.update({("auto", g): len(AUTO_ANIMALS[g]) for g in GENOTYPES})

    for spec, ax in zip(panels, axes.ravel()):
        col = spec["column"]
        if mito_df is None:
            axon_vals = {k: _values(axon_df, col, *k) for k in POS}
            mito_vals = None
        else:
            per_axon = mito_df.groupby(["genotype", "pipeline", "sample", "axon_uid"],
                                       as_index=False)[col].median()
            axon_vals = {k: _values(per_axon, col, *k) for k in POS}
            mito_vals = {k: _values(mito_df, col, *k) for k in POS}

        test_level = spec.get("test_level", "axon")
        test_vals = mito_vals if test_level == "pooled" else axon_vals
        # One value per animal: the median of whatever unit this panel tests. The animal is
        # the only independent unit here; its p-value goes to the stats CSV, not the plot.
        src = mito_df if (mito_df is not None and test_level == "pooled") else (
            per_axon if mito_df is not None else axon_df)
        per_animal = src.groupby(["genotype", "pipeline", "sample"], as_index=False)[col].median()
        animal_vals = {k: _values(per_animal, col, *k) for k in POS}
        stats, animal_stats = panel4(ax, spec, axon_vals, mito_vals, test_vals,
                                     animal_vals, manual_animals)

        for (kind, key), (u, p) in stats.items():
            if kind == "genotype":
                a, b = ("WT", key), ("KO", key)
                label = f"WT vs KO within {PIPELINE_LABELS[key]}"
            else:
                a, b = (key, "manual"), (key, "auto")
                label = f"manual vs automatic within {key}"
            rows_out.append(dict(
                figure=figure_name, panel=spec["panel"], metric=col, comparison=label,
                level="mito" if mito_vals is not None else "axon", test_level=test_level,
                source_manual=tables["manual"], source_auto=tables["auto"],
                n_a_animals=n_animals[(a[1], a[0])], n_b_animals=n_animals[(b[1], b[0])],
                n_a_axons=len(axon_vals[a]), n_b_axons=len(axon_vals[b]),
                n_a_objects=len(mito_vals[a]) if mito_vals else "",
                n_b_objects=len(mito_vals[b]) if mito_vals else "",
                median_a=float(np.median(test_vals[a])), median_b=float(np.median(test_vals[b])),
                mwu_u=u, p_value=p,
                # the same contrast with each animal reduced to one value; blank for the
                # manual-vs-automatic rows, which are not a genotype contrast
                p_value_per_animal=(animal_stats[key][0] if kind == "genotype" else ""),
                drawn=(kind == "genotype"),
            ))

    for ax in axes.ravel()[len(panels):]:   # 5 panels in a 2x3 grid leaves one empty box
        ax.set_visible(False)

    handles = legend_handles(manual_animals, mito_df is not None)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--qc", action="store_true",
                    help="use mito_morphometrics_3d_qc.csv instead of the raw table")
    ap.add_argument("--imod-variant", default="z25", choices=("z25", "full"),
                    help="z25 matches the automatic 25 um z-window; full is the whole "
                         "traced extent (36-39 um) and so compares unequal z coverage")
    ap.add_argument("--manual-animals", default="matched", choices=("matched", "all"),
                    help="matched: only 4010/4007, the animals the automatic pipeline also "
                         "covers, so a difference is the pipeline. all: the whole IMOD set, "
                         "3 animals per genotype -- the automatic side still has 1 each, so "
                         "the two sides are then NOT matched samples")
    ap.add_argument("--png", action="store_true", help="also write PNG alongside the PDF")
    args = ap.parse_args()

    mito_df, axon_df, tables = load_both(args.data_dir, args.qc, args.imod_variant,
                                         args.manual_animals)
    print(f"mode: {'qc' if args.qc else 'raw'} | manual variant: {args.imod_variant} "
          f"| manual animals: {args.manual_animals}")
    if args.manual_animals == "all":
        print("  NOTE manual and automatic are not matched samples: the automatic side is "
              "one animal per genotype")
    for (g, p), n in mito_df.groupby(["genotype", "pipeline"]).size().items():
        print(f"  {g:2s} {p:6s}: {n:4d} mitochondria")

    rows = []
    # Every non-default mode lands in the filename: raw and QC runs of the same figure
    # used to overwrite each other silently, with only a column inside the stats CSV to
    # tell them apart.
    suffix = "" if args.manual_animals == "matched" else "_all_animals"
    if args.qc:
        suffix += "_qc"
    figs = [
        (build_figure(MITO_PANELS, rows, "manual_vs_auto_mito_level", axon_df, mito_df,
                      tables, 2, 3, (13.5, 9), args.manual_animals),
         f"figure_manual_vs_auto_mito_level{suffix}"),
        (build_figure(AXON_PANELS, rows, "manual_vs_auto_axon_level", axon_df, None,
                      tables, 2, 3, (13.5, 9), args.manual_animals),
         f"figure_manual_vs_auto_axon_level{suffix}"),
    ]

    plot_dir = os.path.join(args.out_dir, "Plots")
    table_dir = os.path.join(args.out_dir, "table")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    outputs = []
    for fig, name in figs:
        path = os.path.join(plot_dir, f"{name}.pdf")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        if args.png:
            fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
            outputs.append(path.replace(".pdf", ".png"))
        plt.close(fig)
        outputs.append(path)

    stats = pd.DataFrame(rows)
    stats.insert(0, "mode", "qc" if args.qc else "raw")
    stats.insert(1, "imod_variant", args.imod_variant)
    stats.insert(2, "manual_animals", args.manual_animals)
    stats_path = os.path.join(table_dir, f"manual_vs_auto_statistics{suffix}.csv")
    stats.to_csv(stats_path, index=False)
    outputs.append(stats_path)

    print("\ngenotype effect, drawn on the figures:")
    drawn = stats[stats.drawn]
    for _, r in drawn.iterrows():
        print(f"  {r.figure.rsplit('_', 2)[-2]:5s} {r.panel} {r.metric:26s} {r.comparison:32s} "
              f"WT {r.median_a:9.4f}  KO {r.median_b:9.4f}  p={r.p_value:.3f}")
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
