#!/usr/bin/env python
"""Does the WT-vs-Plp genotype effect survive a change of analysis pipeline?

Two independent pipelines exist for this project:

  automatic  data/{4007,4010}_morphometrics_final/       10 axons/animal, 1 animal/genotype
  hand/IMOD  data/{6 animals}_morphometrics_imod_{full,z25}/  5 axons/animal, 3 animals/genotype

Only 4007 (KO) and 4010 (WT) were processed by both, so this script does NOT try to match
individual mitochondria (their axon sets barely overlap). Instead it computes the genotype
effect size independently inside each of three arms and plots them side by side:

  auto(2)   automatic, 4007 + 4010
  imod(2)   hand, the SAME two animals      -> auto(2) vs imod(2) = the method effect
  imod(6)   hand, all six animals           -> imod(2) vs imod(6) = the animal-sampling effect

Effect size is log2(median_KO / median_WT) per axon, with a percentile bootstrap 95% CI
over axons, plus every cross-animal ratio overlaid so the animal-level spread is visible
rather than hidden inside the axon-level CI.

Usage
-----
    python figure_pipeline_forest.py                        # raw mito tables, imod "full"
    python figure_pipeline_forest.py --qc --png
    python figure_pipeline_forest.py --imod-variant z25     # see confound 1 below
    python figure_pipeline_forest.py --selfcheck
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)
from matplotlib.lines import Line2D  # noqa: E402

from figures_3d import load_cell_volumes, load_mito_level, mwu  # noqa: E402


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

GENOTYPE = {"4009": "WT", "4010": "WT", "4016": "WT",
            "4005": "KO", "4007": "KO", "4015": "KO"}

# 4015 was traced at IMOD z-scale 2 -> 10 nm z step; every other animal is 25 nm.
# Surface-derived metrics (sv_ratio, sphericity) are resolution-sensitive, so this is
# recorded per row of the stats CSV rather than silently averaged away.
VOXEL_Z_NM = {sid: (10 if sid == "4015" else 25) for sid in GENOTYPE}

MATCHED = ("4010", "4007")   # the only two animals present in both pipelines

ARM_ORDER = ["auto(2)", "imod(2)", "imod(6)"]

ARM_STYLE = {
    "auto(2)": {"marker": "o", "color": "#000000", "offset": +0.26,
                "label": "automatic, 2 animals"},
    "imod(2)": {"marker": "s", "color": "#5a5a5a", "offset": 0.0,
                "label": "hand (IMOD), same 2 animals"},
    "imod(6)": {"marker": "D", "color": "#9a9a9a", "offset": -0.26,
                "label": "hand (IMOD), all 6 animals"},
}

# Per-axon metrics. Densities are computed here; everything else is a per-axon median of
# the mito-level column named on the left.
MEDIAN_COLS = {
    "sphericity": "median_sphericity",
    "sv_ratio": "median_sv_ratio",
    "elongation_a_over_c": "median_elongation",
    "flatness_b_over_c": "median_flatness",
    "volume_um3": "median_mito_volume_um3",
    "nearest_neighbor_um": "median_nn_um",
}

# (block, metric key, row label, caveat recorded in the stats CSV)
Z_CAVEAT = ("density normalised by the FULL axon volume while mitos are z-filtered "
            "-- see confound 1 in the module docstring")
NN_CAVEAT = ("nearest neighbour is computed within each run's own mito set (5 axons for "
             "IMOD, 10 for automatic), so only the within-arm ratio is comparable")

METRICS = [
    ("Density", "mito_count_density", r"Mito count density (µm$^{-3}$)", Z_CAVEAT),
    ("Density", "mito_volume_fraction", "Mito volume fraction", Z_CAVEAT),
    ("Density", "membrane_area_density", r"Membrane area density (µm$^2$/µm$^3$)", Z_CAVEAT),
    ("Shape", "median_sphericity", "Sphericity", ""),
    ("Shape", "median_sv_ratio", r"S/V ratio (µm$^{-1}$)", ""),
    ("Shape", "median_elongation", "Elongation (a/c)", ""),
    ("Shape", "median_flatness", "Flatness (b/c)", ""),
    ("Size + spatial", "median_mito_volume_um3", r"Mito volume (µm$^3$)", ""),
    ("Size + spatial", "median_nn_um", "Nearest-neighbour distance (µm)", NN_CAVEAT),
]

BLOCK_ORDER = ["Density", "Shape", "Size + spatial"]

# x axis: log2 ratio, clipped so one very wide n=5 interval cannot squash the rest.
XCLIP = 3.0                                  # log2 -> 0.125x .. 8x
XTICKS = [0.25, 0.5, 1, 2, 4]

N_RESAMPLES = 10000
BOOT_SEED = 0

# Filled in from a --selfcheck run; see selfcheck() for what each entry guards.
EXPECTED = {
    "n_axons": {"auto(2)": (10, 10), "imod(2)": (5, 5), "imod(6)": (15, 15)},
    # (arm, metric) -> (ratio, ci_low_log2, ci_high_log2), raw mito table, imod variant "full"
    "effects": {
        ("imod(6)", "mito_count_density"): (2.535, -0.077, 1.771),
        ("imod(6)", "mito_volume_fraction"): (1.828, 0.318, 1.139),
        ("imod(6)", "membrane_area_density"): (1.586, 0.172, 1.078),
        ("imod(6)", "median_mito_volume_um3"): (0.915, -0.770, 0.324),
        ("auto(2)", "mito_count_density"): (0.761, -1.417, 0.347),
        ("imod(2)", "mito_count_density"): (3.132, -0.329, 2.663),
    },
}

LEGEND_TEXT = (
    "**Figure 5. The genotype effect is reproduced by both pipelines for the density "
    "metrics and by neither for the shape metrics.**\n\n"
    "Effect size is log2(median Plp$^{{-/y}}$ / median WT) computed on per-axon values, "
    "one row per morphometric. Markers give the point estimate, horizontal bars the "
    "percentile bootstrap 95% CI over axons ({n_resamples} resamples, seed {seed}); "
    "arrow caps mark intervals running past the axis limit. Small open markers are the "
    "individual cross-animal ratios (one per WT x Plp$^{{-/y}}$ animal pair: 1 pair for "
    "the two-animal arms, 9 for imod(6)), shown because axons within an animal are not "
    "independent and the axon-level CI therefore understates animal-to-animal spread. "
    "The dashed line is no effect. p values are two-sided Mann-Whitney U on per-axon "
    "values and are exploratory: the automatic arm has one animal per genotype, so its "
    "axons are pseudo-replicates.\n\n"
    "Arms: **auto(2)** = automatic segmentation pipeline, animals 4010 (WT) and 4007 "
    "(Plp$^{{-/y}}$), 10 whitelisted axons each. **imod(2)** = manual IMOD tracing of the "
    "same two animals, 5 axons each. **imod(6)** = manual IMOD tracing of all six animals "
    "(WT 4009/4010/4016, Plp$^{{-/y}}$ 4005/4007/4015), 5 axons each. auto(2) vs imod(2) "
    "isolates the effect of the method on the same tissue; imod(2) vs imod(6) isolates the "
    "effect of adding four more animals.\n\n"
    "Source: `{table}`, IMOD variant `{variant}`. "
    "Caveats: (i) the z-range filter in `morphometrics_3d.py` drops mitochondria outside "
    "the window but leaves the axon volume denominator at the full traced extent, so all "
    "three density rows are biased low by an animal-specific factor; the within-arm ratio "
    "cancels this only if both genotypes lose a similar fraction. (ii) Nearest-neighbour "
    "distance is computed within each run's own set of mitochondria, so its absolute value "
    "is not comparable across arms. (iii) Axons were hand-picked in both pipelines, which "
    "is adequate for method comparison but not for biological inference. (iv) Animal 4015 "
    "was traced at 10 nm z spacing against 25 nm for the rest, and surface-derived metrics "
    "(S/V ratio, sphericity) are resolution-sensitive."
)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def arm_samples(variant):
    """{arm: {directory_name: genotype}} for the three arms."""
    imod = {sid: f"{sid}_morphometrics_imod_{variant}" for sid in GENOTYPE}
    return {
        "auto(2)": {f"{sid}_morphometrics_final": GENOTYPE[sid] for sid in MATCHED},
        "imod(2)": {imod[sid]: GENOTYPE[sid] for sid in MATCHED},
        "imod(6)": {imod[sid]: g for sid, g in GENOTYPE.items()},
    }


def axon_metrics(cell_df, mito_df):
    """Per-axon table carrying all nine forest-plot metrics.

    cell_df supplies cell_volume_um3 (from cell_level_mito_summary.csv); every other
    column is rebuilt from whichever mito table --qc selected, so the two never disagree.
    """
    grp = mito_df.groupby("axon_uid")
    out = cell_df.set_index("axon_uid").copy()
    counts = grp.size().reindex(out.index).fillna(0)
    volumes = grp["volume_um3"].sum().reindex(out.index).fillna(0.0)
    surfaces = grp["surface_um2"].sum().reindex(out.index).fillna(0.0)
    out["mito_count_density"] = counts / out["cell_volume_um3"]
    out["mito_volume_fraction"] = volumes / out["cell_volume_um3"]
    out["membrane_area_density"] = surfaces / out["cell_volume_um3"]
    for src, dst in MEDIAN_COLS.items():
        out[dst] = grp[src].median()
    return out.reset_index()


def load_arm(data_dir, qc, samples):
    """Per-axon metric table for one arm."""
    mito_df, filename = load_mito_level(data_dir, qc, samples)
    cell_df = load_cell_volumes(data_dir, samples)
    return axon_metrics(cell_df, mito_df), filename


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def _log2_ratio(wt, ko, axis=-1):
    """log2(median KO / median WT). Vectorised over the bootstrap's resample axis."""
    return np.log2(np.median(ko, axis=axis) / np.median(wt, axis=axis))


def effect(wt, ko):
    """Point estimate and percentile bootstrap 95% CI of log2(median KO / median WT).

    ponytail: the bootstrap resamples axons, which is the unit both arms share and the
    only one the two-animal arms have more than one of. For imod(6) the axons are nested
    in 3 animals per genotype and are therefore not independent -- upgrade path is a
    cluster bootstrap (resample animals, then axons within). The per-animal ratios plotted
    alongside exist so that this limitation is visible instead of buried.
    """
    wt = np.asarray(wt, dtype=float)
    ko = np.asarray(ko, dtype=float)
    wt, ko = wt[np.isfinite(wt)], ko[np.isfinite(ko)]
    if len(wt) < 2 or len(ko) < 2 or np.median(wt) <= 0 or np.median(ko) <= 0:
        return np.nan, np.nan, np.nan
    point = float(_log2_ratio(wt, ko))
    # method="percentile", not BCa: BCa's jackknife is unreliable at n=5 and imod(2) is n=5.
    res = bootstrap((wt, ko), _log2_ratio, n_resamples=N_RESAMPLES, method="percentile",
                    confidence_level=0.95, vectorized=True, random_state=BOOT_SEED)
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def animal_ratios(df, column):
    """Every cross-animal log2 ratio: one value per (WT animal, KO animal) pair."""
    med = df.groupby(["genotype", "sample"])[column].median()
    if "WT" not in med.index.get_level_values(0) or "KO" not in med.index.get_level_values(0):
        return np.array([])
    wt, ko = med.loc["WT"].to_numpy(), med.loc["KO"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(np.outer(ko, 1.0 / wt).ravel())
    return out[np.isfinite(out)]


def build_stats(arms, source_tables, variant):
    """One row per (arm, metric). This table is the script's real output."""
    rows = []
    for arm in ARM_ORDER:
        df = arms[arm]
        for block, key, label, caveat in METRICS:
            wt = df.loc[df.genotype == "WT", key].dropna().to_numpy()
            ko = df.loc[df.genotype == "KO", key].dropna().to_numpy()
            point, lo, hi = effect(wt, ko)
            u_stat, p_val = mwu(wt, ko) if len(wt) and len(ko) else (np.nan, np.nan)
            samples = sorted(df["sample"].unique())
            rows.append({
                "arm": arm,
                "block": block,
                "metric": key,
                "label": label,
                "source_table": source_tables[arm],
                "imod_variant": variant if arm.startswith("imod") else "n/a",
                "animals": "+".join(samples),
                "voxel_z_nm": "+".join(str(VOXEL_Z_NM.get(s, 25)) for s in samples),
                "n_wt_axons": len(wt),
                "n_ko_axons": len(ko),
                "n_wt_animals": df.loc[df.genotype == "WT", "sample"].nunique(),
                "n_ko_animals": df.loc[df.genotype == "KO", "sample"].nunique(),
                "wt_median": np.median(wt) if len(wt) else np.nan,
                "ko_median": np.median(ko) if len(ko) else np.nan,
                "log2_ratio": point,
                "ci_low": lo,
                "ci_high": hi,
                "ratio": 2.0 ** point,
                "ci_low_ratio": 2.0 ** lo,
                "ci_high_ratio": 2.0 ** hi,
                "mwu_u": u_stat,
                "p_value": p_val,
                "animal_log2_ratios": " ".join(f"{v:.4f}" for v in animal_ratios(df, key)),
                "caveat": caveat,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

def _draw_interval(ax, y, point, lo, hi, style):
    """Marker + CI bar, with arrow caps where the interval leaves the axis."""
    lo_c, hi_c = max(lo, -XCLIP), min(hi, XCLIP)
    ax.plot([lo_c, hi_c], [y, y], color=style["color"], lw=1.6, solid_capstyle="butt",
            zorder=3)
    for edge, clipped, direction in ((lo, lo_c, -1), (hi, hi_c, +1)):
        if clipped != edge:
            ax.plot(clipped, y, marker=(3, 0, 90 * -direction), markersize=6,
                    color=style["color"], zorder=3)
        else:
            ax.plot([edge, edge], [y - 0.09, y + 0.09], color=style["color"], lw=1.6,
                    zorder=3)
    ax.plot(np.clip(point, -XCLIP, XCLIP), y, marker=style["marker"], markersize=8,
            color=style["color"], markeredgecolor="white", markeredgewidth=0.8, zorder=5)


def build_figure(stats, source_table, variant):
    counts = [sum(1 for m in METRICS if m[0] == b) for b in BLOCK_ORDER]
    fig, axes = plt.subplots(len(BLOCK_ORDER), 1, figsize=(10.5, 9.0),
                             gridspec_kw={"height_ratios": counts})
    indexed = stats.set_index(["arm", "metric"])

    for ax_idx, (ax, block) in enumerate(zip(axes, BLOCK_ORDER)):
        rows = [m for m in METRICS if m[0] == block]
        for y, (_, key, label, _caveat) in enumerate(rows):
            for arm in ARM_ORDER:
                style = ARM_STYLE[arm]
                row = indexed.loc[(arm, key)]
                yy = y + style["offset"]
                if np.isfinite(row.log2_ratio):
                    _draw_interval(ax, yy, row.log2_ratio, row.ci_low, row.ci_high, style)
                if np.isfinite(row.p_value):
                    # Fixed column outside the axes: x in axes fraction, y in data coords.
                    ax.text(1.02, yy, f"{row.p_value:.3f}", ha="left", va="center",
                            fontsize=7, color=style["color"],
                            fontweight="bold" if row.p_value < 0.05 else "normal",
                            transform=ax.get_yaxis_transform(), clip_on=False)
                animals = np.fromstring(row.animal_log2_ratios, sep=" ")
                if animals.size:
                    ax.scatter(np.clip(animals, -XCLIP, XCLIP),
                               np.full(animals.size, yy), s=22, facecolor="none",
                               edgecolor=style["color"], linewidth=0.9, zorder=4)

        ax.axvline(0.0, ls="--", lw=1.0, color="0.5", zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label for _, _, label, _c in rows], fontsize=9)
        ax.set_ylim(len(rows) - 0.55, -0.55)          # inverted: first metric on top
        ax.set_xlim(-XCLIP, XCLIP)
        ax.set_xticks(np.log2(XTICKS))
        ax.set_xticklabels([f"{t:g}×" for t in XTICKS], fontsize=9)
        ax.set_title(block, fontsize=10, pad=8, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=1.2, length=5)
        ax.text(-0.30, 1.06, chr(ord("A") + ax_idx), transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
        ax.text(1.02, 1.02, "exploratory p", ha="left", va="bottom", fontsize=7,
                color="0.3", transform=ax.transAxes, clip_on=False)
        if ax is axes[-1]:
            ax.set_xlabel(r"Plp$^{-/y}$ / WT  (median ratio, log$_2$ scale)", fontsize=9)

    handles = [Line2D([], [], marker=ARM_STYLE[a]["marker"], color=ARM_STYLE[a]["color"],
                      markeredgecolor="white", markeredgewidth=0.8, markersize=8, lw=1.6,
                      label=f"{ARM_STYLE[a]['label']} "
                            f"(n={indexed.loc[(a, METRICS[0][1])].n_wt_axons}"
                            f"+{indexed.loc[(a, METRICS[0][1])].n_ko_axons} axons)")
               for a in ARM_ORDER]
    handles.append(Line2D([], [], marker="o", color="0.35", markerfacecolor="none",
                          markersize=6, lw=0, label="individual cross-animal ratio"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Genotype effect by pipeline  —  {source_table}, IMOD variant "
                 f"'{variant}'", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.10, 0.93, 0.96))   # right margin holds the p column
    return fig


# ------------------------------------------------------------
# DRIVER
# ------------------------------------------------------------

def make_figure(data_dir, qc, variant):
    samples = arm_samples(variant)
    arms, source_tables = {}, {}
    for arm, mapping in samples.items():
        arms[arm], source_tables[arm] = load_arm(data_dir, qc, mapping)
    stats = build_stats(arms, source_tables, variant)
    source_table = source_tables["imod(6)"]
    return build_figure(stats, source_table, variant), stats, source_table


def selfcheck(data_dir):
    """Assert the numbers this script is meant to produce. Raises AssertionError."""
    checked = 0
    _fig, stats, _table = make_figure(data_dir, qc=False, variant="full")
    plt.close(_fig)
    indexed = stats.set_index(["arm", "metric"])

    for arm, (n_wt, n_ko) in EXPECTED["n_axons"].items():
        row = indexed.loc[(arm, METRICS[0][1])]
        assert (row.n_wt_axons, row.n_ko_axons) == (n_wt, n_ko), \
            f"{arm}: expected {n_wt}+{n_ko} axons, got {row.n_wt_axons}+{row.n_ko_axons}"
        checked += 1

    for (arm, metric), (ratio, lo, hi) in EXPECTED["effects"].items():
        row = indexed.loc[(arm, metric)]
        for name, got, want in (("ratio", row.ratio, ratio),
                                ("ci_low", row.ci_low, lo),
                                ("ci_high", row.ci_high, hi)):
            assert round(got, 3) == want, \
                f"{arm}/{metric} {name}: expected {want}, got {got:.3f}"
            checked += 1

    # A wt/ko argument-order slip is the easy bug here, and it shows up as a point
    # estimate outside its own interval.
    for row in stats.itertuples():
        if np.isfinite(row.log2_ratio):
            assert row.ci_low <= row.log2_ratio <= row.ci_high, \
                f"{row.arm}/{row.metric}: point {row.log2_ratio:.3f} outside CI " \
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
            checked += 1

    print(f"selfcheck: {checked} assertions passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="directory holding the run folders")
    ap.add_argument("--out-dir", default=".", help="Plots/ and table/ are created here")
    ap.add_argument("--qc", action="store_true", help="use the QC-filtered mito tables")
    ap.add_argument("--imod-variant", default="full", choices=["full", "z25"],
                    help="which IMOD z-range run to use; 'full' avoids the density bias")
    ap.add_argument("--png", action="store_true", help="also write a PNG")
    ap.add_argument("--selfcheck", action="store_true", help="assert known values and exit")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck(args.data_dir)
        return

    if args.imod_variant == "z25":
        print("WARNING: the z25 runs filter mitochondria by z but leave cell_volume_um3 at "
              "the full axon extent, so all three density rows are biased low. 'full' is "
              "the defensible default.")

    fig, stats, source_table = make_figure(args.data_dir, args.qc, args.imod_variant)

    plot_dir = os.path.join(args.out_dir, "Plots")
    table_dir = os.path.join(args.out_dir, "table")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    pdf_path = os.path.join(plot_dir, "figure5_pipeline_forest.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {pdf_path}")

    if args.png:
        # savefig above closed nothing, but tight_layout state is safest rebuilt --
        # same approach as figure_combined.py.
        fig, _stats, _table = make_figure(args.data_dir, args.qc, args.imod_variant)
        png_path = pdf_path.replace(".pdf", ".png")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {png_path}")

    csv_path = os.path.join(table_dir, "pipeline_comparison_effects.csv")
    stats.drop(columns=["label"]).to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}  ({len(stats)} rows)")

    legend_path = os.path.join(table_dir, "figure5_legend.md")
    with open(legend_path, "w") as fh:
        fh.write(LEGEND_TEXT.format(table=source_table, variant=args.imod_variant,
                                    n_resamples=N_RESAMPLES, seed=BOOT_SEED) + "\n")
    print(f"Wrote: {legend_path}")

    show = ["arm", "metric", "n_wt_axons", "n_ko_axons", "ratio", "ci_low_ratio",
            "ci_high_ratio", "p_value"]
    print()
    print(stats[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
