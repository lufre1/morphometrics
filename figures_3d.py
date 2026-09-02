#!/usr/bin/env python
"""Build the two result figures from the 3D morphometrics CSVs in data/.

Figure 1 (axon level)  : median elongation, median flatness, mito volume density,
                         mito count density -- one point per axon (n = 10 per genotype).
Figure 2 (mito level)   : volume, sphericity, surface-to-volume ratio, nearest-neighbour
                         distance, elongation, flatness -- translucent cloud of individual
                         mitochondria plus large per-axon medians (superplot). Elongation
                         and flatness are tested pooled over all mitochondria; the other
                         four are tested on per-axon medians.

Both figures are derived from the SAME mitochondria table: the raw one by default, the
QC'd one with --qc. Figure 1's quantities are recomputed here rather than read from the
median_*/`*_density` columns of cell_level_mito_summary.csv -- those columns were computed
from the QC'd table, so reading them made figure 1 QC-based while figure 2 was raw-based.
The only column still taken from that CSV is cell_volume_um3.

Consequence: the originally published figure 1 A/B p-values (0.850 / 0.054) reproduce
under --qc, not in the default raw mode (0.734 / 0.076). --selfcheck asserts both modes.

Plot styling follows the house style of the 2D scripts in RGC/, LGN/ and N.opt./
(violin outline + narrow box + jittered points, Dark2/Set2 genotype colours).

Usage
-----
    python figures_3d.py                      # both figures from the raw mito table
    python figures_3d.py --qc                 # both figures from the QC'd mito table
    python figures_3d.py --selfcheck          # assert the p-values of both modes still hold
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)
from matplotlib.lines import Line2D  # noqa: E402


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

WT_COLOR = "#1b9e77"
KO_COLOR = "#fc8d62"
GROUP_LABELS = ["WT", r"$\mathit{Plp}^{-/y}$"]

# ponytail: genotype lives only in the directory name -- none of the CSVs carry a
# genotype column. If more samples arrive, promote this to a --genotype CLI mapping.
SAMPLES = {
    "4010_morphometrics_final": "WT",
    "4007_morphometrics_final": "KO",
}

# Expected p-values per mode, used by --selfcheck. The "qc" column reproduces the
# originally published figure 1; "raw" is what the default (aligned) mode gives.
EXPECTED_P = {
    "raw": {
        "F1A": 0.734, "F1B": 0.076, "F1C": 0.064, "F1D": 0.345,
        "F2A": 0.121, "F2B": 0.345, "F2C": 0.002, "F2D": 0.678,
        "F2E": 0.217, "F2F": 0.073,
    },
    "qc": {
        "F1A": 0.850, "F1B": 0.054, "F1C": 0.021, "F1D": 0.427,
        "F2A": 0.089, "F2B": 0.734, "F2C": 0.002, "F2D": 0.734,
        "F2E": 0.390, "F2F": 0.035,
    },
}


# ponytail: y-limits are hardcoded so the output matches the published figures.
# Switch to (0, 1.15 * max) if new data overflows -- --selfcheck catches overflow.
FIG1_PANELS = [
    dict(panel="F1A", letter="A",
         title="3D median mitochondrial elongation per axon",
         ylabel="Median mitochondrial elongation (a/c)",
         column="median_mito_elongation",
         ylim=(0, 12), ytick_step=2),
    dict(panel="F1B", letter="B",
         title="3D median mitochondrial flatness per axon",
         ylabel="Median mitochondrial flatness (b/c)",
         column="median_mito_flatness",
         ylim=(0, 2.5), ytick_step=0.5),
    dict(panel="F1C", letter="C",
         title="3D axonal mitochondrial volume density",
         ylabel="Mitochondrial volume density (µm³ / 100 µm³ axon)",
         column="mito_volume_density_per100",
         ylim=(0, 20), ytick_step=2),
    dict(panel="F1D", letter="D",
         title="3D axonal mitochondrial count density",
         ylabel="Mitochondrial density (mitochondria / 100 µm³)",
         column="mito_count_density_per100",
         ylim=(0, 140), ytick_step=20),
]

FIG2_PANELS = [
    dict(panel="F2A", letter="A",
         title="3D mitochondrial volume",
         ylabel="Mitochondrial volume (µm³)",
         column="volume_um3",
         ylim=(0, 0.9), ytick_step=0.1),
    dict(panel="F2B", letter="B",
         title="3D mitochondrial sphericity",
         ylabel="Mitochondrial sphericity",
         column="sphericity",
         ylim=(0, 1.0), ytick_step=0.2),
    dict(panel="F2C", letter="C",
         title="3D individual mitochondrial surface-to-volume ratio",
         ylabel="Individual mitochondrial surface-to-volume ratio (µm⁻¹)",
         column="sv_ratio",
         ylim=(0, 120), ytick_step=20),
    dict(panel="F2D", letter="D",
         title="3D mitochondrial nearest-neighbor distance",
         ylabel="Nearest-neighbor distance (µm)",
         column="nearest_neighbor_um",
         ylim=(0, 5.0), ytick_step=0.5),
    # Elongation and flatness are tested pooled over all mitochondria, not on per-axon
    # medians: mitochondrial shape varies too much within one axon for a median to mean
    # anything. ylim covers both tables -- raw elongation reaches 27.14 (the QC filter is
    # elongation >= 20, so the raw table legitimately holds higher values), QC reaches 19.95.
    dict(panel="F2E", letter="E",
         title="3D mitochondrial elongation",
         ylabel="Mitochondrial elongation (a/c)",
         column="elongation_a_over_c",
         ylim=(0, 30), ytick_step=5, test_level="pooled", p_suffix=" (pooled)"),
    dict(panel="F2F", letter="F",
         title="3D mitochondrial flatness",
         ylabel="Mitochondrial flatness (b/c)",
         column="flatness_b_over_c",
         ylim=(0, 7), ytick_step=1, test_level="pooled", p_suffix=" (pooled)"),
]


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def _load_concat(data_dir, filename):
    """Concatenate one CSV per sample, tagging each row with sample and genotype."""
    frames = []
    for sample, genotype in SAMPLES.items():
        path = os.path.join(data_dir, sample, filename)
        df = pd.read_csv(path)
        df["sample"] = sample.split("_")[0]
        df["genotype"] = genotype
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _axon_uid(df):
    """cell_id repeats across samples (e.g. 106 exists in both), so key axons uniquely.

    Same trick as N.opt./Sirt2-deficient/density:121.
    """
    return df["sample"] + "__axon_" + df["cell_id"].astype(int).astype(str)


def load_cell_volumes(data_dir):
    """Axon volumes -- the only thing still read from cell_level_mito_summary.csv.

    Its summary columns (mito_count, *_density, median_mito_*) are all derived from the
    QC'd mito table, so using them would pin figure 1 to QC data regardless of --qc.
    """
    df = _load_concat(data_dir, "cell_level_mito_summary.csv")
    df["axon_uid"] = _axon_uid(df)
    return df[["axon_uid", "sample", "genotype", "cell_volume_um3"]]


def load_mito_level(data_dir, qc):
    """Per-mitochondrion table. qc=True uses the QC-filtered variant."""
    filename = "mito_morphometrics_3d_qc.csv" if qc else "mito_morphometrics_3d.csv"
    df = _load_concat(data_dir, filename)
    df["axon_uid"] = _axon_uid(df)
    return df, filename


def build_axon_table(cell_df, mito_df):
    """Per-axon table for figure 1, recomputed from whichever mito table is in use.

    Reproduces cell_level_mito_summary.csv exactly (to float precision) when mito_df is
    the QC'd table -- checked for mito_count, both densities and the median columns.
    """
    grp = mito_df.groupby("axon_uid")
    out = cell_df.set_index("axon_uid").copy()
    counts = grp.size().reindex(out.index).fillna(0)
    volumes = grp["volume_um3"].sum().reindex(out.index).fillna(0.0)
    out["median_mito_elongation"] = grp["elongation_a_over_c"].median()
    out["median_mito_flatness"] = grp["flatness_b_over_c"].median()
    # The figure y-axes are per 100 µm³ of axon.
    out["mito_count_density_per100"] = 100.0 * counts / out["cell_volume_um3"]
    out["mito_volume_density_per100"] = 100.0 * volumes / out["cell_volume_um3"]
    return out.reset_index()


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def mwu(wt_vals, ko_vals):
    """Two-sided Mann-Whitney U on per-axon values.

    ponytail: plain MWU rather than the GEE/Exchangeable model the 2D scripts use.
    There is one animal per genotype here, so there is no cluster level to model and
    the axons are pseudo-replicates -- hence the annotation says "exploratory".
    """
    res = mannwhitneyu(wt_vals, ko_vals, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

def add_bracket(ax, x1, x2, y, h, text):
    """Significance bracket with a numeric p-value label above it."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.3, color="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


def panel(ax, spec, axon_vals, mito_vals=None, test_vals=None, jitter_seed=0):
    """Draw one violin/box/strip panel and return (u_statistic, p_value).

    axon_vals -- (wt, ko) arrays of per-axon values, drawn as the large white-ringed dots.
                 Suppressed when the spec sets show_axon_dots=False.
    mito_vals -- optional (wt, ko) arrays of per-object values. When given, the violin,
                 box and small translucent dots describe the per-object distribution.
    test_vals -- (wt, ko) arrays the Mann-Whitney U runs on. Defaults to axon_vals; pass
                 mito_vals for a pooled ("global") test.
    """
    colors = [WT_COLOR, KO_COLOR]
    dist = list(mito_vals) if mito_vals is not None else list(axon_vals)

    parts = ax.violinplot(
        dist, positions=[0, 1], showmeans=False, showmedians=False,
        showextrema=False, widths=0.6, bw_method=0.3, points=200,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.0)

    box = ax.boxplot(dist, positions=[0, 1], widths=0.18, patch_artist=True, showfliers=False)
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
        # Small points: individual mitochondria.
        for x, vals, color in zip([0, 1], mito_vals, colors):
            ax.scatter(rng.normal(x, 0.08, len(vals)), vals,
                       s=10, alpha=0.35, color=color, zorder=2)

    # Large white-ringed points: one per axon. Identical marker in every figure so the
    # symbol always means the same thing; the ring keeps them legible over the small-dot
    # cloud. Drawn in every panel, including the pooled-test ones -- the legend says so.
    if spec.get("show_axon_dots", True):
        for x, vals, color in zip([0, 1], axon_vals, colors):
            ax.scatter(rng.normal(x, 0.05, len(vals)), vals, s=90, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)

    u_stat, p_val = mwu(*(test_vals if test_vals is not None else axon_vals))
    y_top = spec["ylim"][1]
    add_bracket(ax, 0, 1, 0.88 * y_top, 0.02 * y_top,
                f"exploratory p = {p_val:.3f}{spec.get('p_suffix', '')}")

    ax.set_title(spec["title"], fontsize=10, pad=12)
    ax.set_ylabel(spec["ylabel"], fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylim(*spec["ylim"])
    step = spec["ytick_step"]
    ax.set_yticks(np.arange(spec["ylim"][0], y_top + step / 2, step))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=5)
    ax.text(-0.18, 1.06, spec["letter"], transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")

    return u_stat, p_val


def _split(df, column):
    """Return (wt_values, ko_values) for a column, NaNs dropped."""
    return tuple(df.loc[df.genotype == g, column].dropna().to_numpy() for g in ("WT", "KO"))


# Legend proxies are grey: colour encodes genotype, marker size encodes the tier.
# markersize is a diameter in points, scatter's s is an area in points^2 -- sqrt(90) ~ 9.5,
# sqrt(10) ~ 3.2, so these match the real markers.
_LEGEND_STYLES = {
    "mito": dict(markersize=3.2, markerfacecolor="0.35", markeredgecolor="none", alpha=0.55),
    "axon": dict(markersize=9.5, markerfacecolor="0.35", markeredgecolor="white",
                 markeredgewidth=0.8),
}


def _legend_handles(entries):
    """Build proxy artists for a figure-level dot legend from [(tier, label), ...]."""
    return [Line2D([], [], marker="o", linestyle="none", label=label, **_LEGEND_STYLES[tier])
            for tier, label in entries]


def build_figure(panels, rows_out, figure_name, axon_df, mito_df=None, source_table="",
                 nrows=2, ncols=2, figsize=(9, 9), legend_entries=()):
    """Render a panel grid and append one stats row per panel to rows_out."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for spec, ax in zip(panels, axes.ravel()):
        col = spec["column"]
        if mito_df is None:
            axon_vals = _split(axon_df, col)
            mito_vals = None
        else:
            per_axon = mito_df.groupby(["genotype", "axon_uid"], as_index=False)[col].median()
            axon_vals = _split(per_axon, col)
            mito_vals = _split(mito_df, col)

        test_level = spec.get("test_level", "axon")
        test_vals = mito_vals if test_level == "pooled" else axon_vals

        u_stat, p_val = panel(ax, spec, axon_vals, mito_vals, test_vals)

        overflow = max(np.max(v) for v in (mito_vals or axon_vals))
        rows_out.append(dict(
            figure=figure_name, panel=spec["panel"], metric=col,
            source_table=source_table, level="mito" if mito_vals is not None else "axon",
            test_level=test_level,
            n_wt_axons=len(axon_vals[0]), n_ko_axons=len(axon_vals[1]),
            n_wt_objects=len(mito_vals[0]) if mito_vals else "",
            n_ko_objects=len(mito_vals[1]) if mito_vals else "",
            wt_median=np.median(test_vals[0]), ko_median=np.median(test_vals[1]),
            mwu_u=u_stat, p_value=p_val,
            annotation=f"exploratory p = {p_val:.3f}{spec.get('p_suffix', '')}",
            data_max=overflow, ylim_top=spec["ylim"][1],
        ))

    if legend_entries:
        handles = _legend_handles(legend_entries)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    return fig


# ------------------------------------------------------------
# FIGURE LEGENDS
# ------------------------------------------------------------

# The dot legend inside each figure is deliberately one line per marker tier -- six panels
# leave no room for the full caption. The prose below is exported to a file instead, and the
# per-panel numbers are pulled from the same stats rows the figures were drawn from, so the
# caption cannot drift away from the plot.
LEGEND_TEXT = {
    "figure1_axon_level": (
        "**Axon-level 3D mitochondrial morphometrics.** Each large white-ringed dot is one "
        "axon; the violin and box describe those same per-axon values (nothing else is "
        "plotted, so violin, box and dots all show the same {n_axons} numbers per genotype). "
        "All four quantities are recomputed from {table}: **A** and **B** are the median over "
        "every mitochondrion in that axon, **C** and **D** normalise the axon's total "
        "mitochondrial volume / count by the axon's own volume (cell_volume_um3). Colour "
        "encodes genotype (green WT, orange $Plp^{{-/y}}$), marker size encodes the level "
        "(large ringed = axon). Statistics: two-sided Mann-Whitney U on the per-axon values. "
        "With one animal per genotype the axons are pseudo-replicates rather than independent "
        "samples, so no clustered model is fitted and every p-value is labelled exploratory."
    ),
    "figure2_mito_level": (
        "**Mitochondrion-level 3D morphometrics (superplot).** Small translucent dots without "
        "a ring are individual mitochondria from {table}; the violin and box describe that "
        "per-object distribution. In **A-D** the large white-ringed dots are per-axon medians "
        "({n_axons} axons per genotype) and the Mann-Whitney U is run on those medians, not on "
        "the cloud. **E** and **F** show per-axon medians as large white-ringed dots (same "
        "visual style), but the test is run on pooled mitochondria because mitochondrial shape "
        "varies too much within a single axon for its median to mean anything. The per-axon "
        "medians of those same two metrics are figure 1 **A** and **B**, which is where the two "
        "figures meet. Colour encodes genotype (green WT, orange $Plp^{{-/y}}$), marker size "
        "encodes the level. All p-values are exploratory (one animal per genotype; see figure 1)."
    ),
}


def write_legends(stats, path, mode):
    """Write one markdown caption plus a per-panel number table per figure."""
    table = "raw (mito_morphometrics_3d.csv)" if mode == "raw" else "QC'd (mito_morphometrics_3d_qc.csv)"
    lines = [
        "# Figure legends -- 3D morphometrics",
        "",
        f"Generated by `figures_3d.py` in **{mode}** mode: both figures come from the "
        f"{table} mitochondria table.",
        "",
    ]
    for figure_name, caption in LEGEND_TEXT.items():
        sub = stats[stats.figure == figure_name]
        first = sub.iloc[0]
        # Object counts only exist for the mito-level figure, and they vary per panel
        # (NaNs are dropped, e.g. nearest-neighbour distance is undefined for a lone
        # mitochondrion), so they belong in the table rather than in the caption.
        header = ["Panel", "Metric", "Tested on", "n WT", "n KO"]
        if first.n_wt_objects != "":
            header += ["mitochondria WT", "mitochondria KO"]
        header += ["WT median", "KO median", "Mann-Whitney U", "p (exploratory)"]
        lines += [
            f"## {figure_name}",
            "",
            caption.format(table=f"`{first.source_table}`", n_axons=first.n_wt_axons),
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for row in sub.itertuples():
            pooled = row.test_level == "pooled"
            cells = [
                row.panel[-1], f"`{row.metric}`",
                "all mitochondria (pooled)" if pooled else "per-axon medians",
                str(row.n_wt_objects if pooled else row.n_wt_axons),
                str(row.n_ko_objects if pooled else row.n_ko_axons),
            ]
            if row.n_wt_objects != "":
                cells += [str(row.n_wt_objects), str(row.n_ko_objects)]
            cells += [f"{row.wt_median:.3f}", f"{row.ko_median:.3f}",
                      f"{row.mwu_u:.1f}", f"{row.p_value:.3f}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def make_figures(data_dir, qc):
    """Build both figures. Returns (fig1, fig2, stats DataFrame)."""
    mito_df, mito_table = load_mito_level(data_dir, qc)
    axon_df = build_axon_table(load_cell_volumes(data_dir), mito_df)

    rows = []
    fig1 = build_figure(
        FIG1_PANELS, rows, "figure1_axon_level", axon_df,
        source_table=f"{mito_table} (+ cell_volume_um3 from cell_level_mito_summary.csv)",
        legend_entries=[("axon", "one axon (n = 10 per genotype); violin and box show the "
                                 "same 10 values")],
    )
    fig2 = build_figure(
        FIG2_PANELS, rows, "figure2_mito_level", axon_df, mito_df,
        source_table=mito_table, ncols=3, figsize=(13.5, 9),
        legend_entries=[
            ("mito", "one mitochondrion — violin and box show this distribution"),
            ("axon", "one axon, median of its mitochondria"),
        ],
    )
    stats = pd.DataFrame(rows)
    stats.insert(0, "mode", "qc" if qc else "raw")
    return fig1, fig2, stats


def selfcheck(data_dir):
    checked = 0
    for mode, expected_p in EXPECTED_P.items():
        fig1, fig2, stats = make_figures(data_dir, qc=(mode == "qc"))
        plt.close(fig1)
        plt.close(fig2)

        got = dict(zip(stats.panel, stats.p_value))
        for key, expected in expected_p.items():
            assert round(got[key], 3) == expected, \
                f"[{mode}] {key}: expected p={expected}, got {got[key]:.3f}"
            checked += 1

        for row in stats.itertuples():
            assert row.data_max <= row.ylim_top, \
                f"[{mode}] {row.panel}: data max {row.data_max:.3f} exceeds ylim {row.ylim_top}"

        assert set(stats.n_wt_axons) == {10} and set(stats.n_ko_axons) == {10}, \
            f"[{mode}] expected 10 axons/group"

    # The QC-mode figure 1 must equal cell_level_mito_summary.csv, which is where its
    # columns came from before both figures were put on the same table.
    published = _load_concat(data_dir, "cell_level_mito_summary.csv")
    published["axon_uid"] = _axon_uid(published)
    mito_df, _ = load_mito_level(data_dir, qc=True)
    rebuilt = build_axon_table(load_cell_volumes(data_dir), mito_df).set_index("axon_uid")
    published = published.set_index("axon_uid").reindex(rebuilt.index)
    for rebuilt_col, csv_col, scale in [
        ("median_mito_elongation", "median_mito_elongation", 1.0),
        ("median_mito_flatness", "median_mito_flatness", 1.0),
        ("mito_count_density_per100", "mito_count_density", 100.0),
        ("mito_volume_density_per100", "mito_volume_density", 100.0),
    ]:
        diff = float(np.abs(rebuilt[rebuilt_col] - scale * published[csv_col]).max())
        assert diff < 1e-9, f"QC rebuild of {rebuilt_col} differs from the CSV by {diff:.3g}"

    print(f"selfcheck OK: {checked} expected p-values reproduced across "
          f"{len(EXPECTED_P)} modes, no panel overflow, n=10 per group, "
          "QC rebuild matches cell_level_mito_summary.csv.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="directory holding the per-sample folders")
    parser.add_argument("--out-dir", default=".", help="Plots/ and table/ are created here")
    parser.add_argument("--qc", action="store_true",
                        help="build BOTH figures from mito_morphometrics_3d_qc.csv instead of "
                             "the raw table (this is the mode that reproduces the originally "
                             "published figure 1 A/B p-values)")
    parser.add_argument("--selfcheck", action="store_true",
                        help="assert the expected p-values of both modes still reproduce, "
                             "then exit")
    args = parser.parse_args()

    if args.selfcheck:
        selfcheck(args.data_dir)
        return

    print(f"mode: {'qc' if args.qc else 'raw'} -- both figures use "
          f"{'mito_morphometrics_3d_qc.csv' if args.qc else 'mito_morphometrics_3d.csv'}")
    if not args.qc:
        print("NOTE: figure 1 A/B are recomputed from the raw table, so their p-values differ "
              "from the originally published ones (which were QC-derived). Use --qc for those.")

    fig1, fig2, stats = make_figures(args.data_dir, args.qc)

    plot_dir = os.path.join(args.out_dir, "Plots")
    table_dir = os.path.join(args.out_dir, "table")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    outputs = []
    for fig, name in [(fig1, "figure1_axon_level"), (fig2, "figure2_mito_level")]:
        path = os.path.join(plot_dir, f"{name}.pdf")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    stats_path = os.path.join(table_dir, "figures_3d_statistics.csv")
    stats.drop(columns=["data_max", "ylim_top"]).to_csv(stats_path, index=False)
    outputs.append(stats_path)

    legend_path = os.path.join(table_dir, "figure_legends.md")
    write_legends(stats, legend_path, "qc" if args.qc else "raw")
    outputs.append(legend_path)

    print(stats[["panel", "metric", "wt_median", "ko_median", "p_value"]].to_string(index=False))
    print("\nWritten:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
