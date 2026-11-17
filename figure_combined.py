#!/usr/bin/env python
"""Combine the 3D reconstruction screenshots with the mito-level quant panels into one figure.

Layout: panel A = reconstructions (2 rows genotype x 3 cols view), panel B = the six
figure2_mito_level violin/box panels (relettered B-G so the alphabet runs once, top to
bottom). Both blocks are drawn from the SAME plotting code as the standalone figures
(figures_screenshots.py, figures_3d.py) -- nothing here is rasterized, so panel text
stays vector in the output PDF.

Usage
-----
    python figure_combined.py                          # raw mito table, PDF only
    python figure_combined.py --qc --png
    python figure_combined.py --top-view axons_mitos    # top block: just Axons+Mitochondria,
                                                         # WT vs Plp-/y side by side
    python figure_combined.py --top-view axons_mitos --quant manual_vs_auto \
        --manual-animals all --qc --png   # lower block: manual (IMOD) vs automatic
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures_screenshots import (
    SAMPLES, VIEWS, VIEW_LABELS, load_and_pad, compute_target_size,
    calibrate_scale_bars, draw_scale_bar,
)
from figures_3d import (
    FIG2_PANELS, panel, load_mito_level, load_cell_volumes, build_axon_table,
    _split, _legend_handles,
)
from figure_manual_vs_auto import (
    MITO_PANELS as MVA_PANELS, POS as MVA_POS, panel4, load_both, _values,
    legend_handles as mva_legend_handles,
)


def build_reconstruction_block(fig, block_gs, screenshot_dir, target_size, calib):
    sample_ids = list(SAMPLES.keys())  # 4010 (WT), 4007 (KO) -- row order
    for row_idx, sample_id in enumerate(sample_ids):
        for col_idx, view in enumerate(VIEWS):
            ax = fig.add_subplot(block_gs[row_idx, col_idx])
            img_dir = os.path.join(screenshot_dir, f"{sample_id}_3d")
            image_path = os.path.join(img_dir, f"{sample_id}_{view}_3d.png")
            img, scale = load_and_pad(image_path, target_size, calib[sample_id]["erase_bbox"])
            ax.imshow(img)
            ax.axis("off")
            fraction = calib[sample_id]["bar_px_length"] * scale / target_size[1]
            draw_scale_bar(ax, fraction)

            if row_idx == 0:
                ax.set_title(VIEW_LABELS[view], fontsize=11, pad=8)
            if col_idx == 0:
                ax.text(-0.12, 0.5, SAMPLES[sample_id], transform=ax.transAxes,
                        fontsize=11, va="center", ha="right", rotation=90)

    fig.text(block_gs[0, 0].get_position(fig).x0 - 0.05,
              block_gs[0, 0].get_position(fig).y1 + 0.01,
              "A", fontsize=24, fontweight="bold")


def build_single_view_block(fig, block_gs, screenshot_dir, target_size, calib, view):
    """1 row x 2 cols (genotype) -- for when only one view is wanted on top.

    Row height is sized (in build_combined_figure) so each column fills the full figure
    width edge to edge -- no side letterboxing, no dead space between the two images.
    """
    sample_ids = list(SAMPLES.keys())  # 4010 (WT), 4007 (KO) -- col order
    for col_idx, sample_id in enumerate(sample_ids):
        ax = fig.add_subplot(block_gs[0, col_idx])
        img_dir = os.path.join(screenshot_dir, f"{sample_id}_3d")
        image_path = os.path.join(img_dir, f"{sample_id}_{view}_3d.png")
        img, scale = load_and_pad(image_path, target_size, calib[sample_id]["erase_bbox"])
        ax.imshow(img)
        ax.axis("off")
        fraction = calib[sample_id]["bar_px_length"] * scale / target_size[1]
        draw_scale_bar(ax, fraction)
        ax.set_title(SAMPLES[sample_id], fontsize=20, pad=8)

    fig.text(block_gs[0, 0].get_position(fig).x0 - 0.05,
              block_gs[0, 0].get_position(fig).y1 + 0.01,
              "A", fontsize=24, fontweight="bold")


def build_quant_block(fig, block_gs, axon_df, mito_df):
    # Reletter B-G so the combined figure runs one alphabet: A = images, B-G = quant.
    relettered = [dict(spec, letter=chr(ord("B") + i)) for i, spec in enumerate(FIG2_PANELS)]

    for i, spec in enumerate(relettered):
        ax = fig.add_subplot(block_gs[i // 3, i % 3])
        col = spec["column"]
        per_axon = mito_df.groupby(["genotype", "axon_uid"], as_index=False)[col].median()
        axon_vals = _split(per_axon, col)
        mito_vals = _split(mito_df, col)
        test_vals = mito_vals if spec.get("test_level") == "pooled" else axon_vals
        panel(ax, spec, axon_vals, mito_vals, test_vals)


def build_quant_block_mva(fig, block_gs, mito_df, manual_animals):
    """Lower block = the manual (IMOD) vs automatic mito-level panels, relettered B-G."""
    relettered = [dict(spec, letter=chr(ord("B") + i)) for i, spec in enumerate(MVA_PANELS)]

    for i, spec in enumerate(relettered):
        ax = fig.add_subplot(block_gs[i // 3, i % 3])
        col = spec["column"]
        per_axon = mito_df.groupby(["genotype", "pipeline", "sample", "axon_uid"],
                                   as_index=False)[col].median()
        axon_vals = {k: _values(per_axon, col, *k) for k in MVA_POS}
        mito_vals = {k: _values(mito_df, col, *k) for k in MVA_POS}
        test_vals = mito_vals if spec.get("test_level") == "pooled" else axon_vals
        # one value per animal -- panel4 needs it for its stats, the plot does not show it
        src = mito_df if spec.get("test_level") == "pooled" else per_axon
        per_animal = src.groupby(["genotype", "pipeline", "sample"], as_index=False)[col].median()
        animal_vals = {k: _values(per_animal, col, *k) for k in MVA_POS}
        panel4(ax, spec, axon_vals, mito_vals, test_vals, animal_vals, manual_animals)


def build_combined_figure(screenshot_dir, data_dir, qc, top_view="all",
                          quant="fig2", imod_variant="z25", manual_animals="all",
                          legend=True):
    image_paths = [
        os.path.join(screenshot_dir, f"{sample_id}_3d", f"{sample_id}_{view}_3d.png")
        for sample_id in SAMPLES for view in VIEWS
    ]
    target_size = compute_target_size(image_paths)
    calib = calibrate_scale_bars(screenshot_dir)
    target_h, target_w = target_size

    if quant == "manual_vs_auto":
        mito_df, axon_df, tables = load_both(data_dir, qc, imod_variant, manual_animals)
        mito_table = f"manual {tables['manual']} vs auto {tables['auto']}"
    else:
        mito_df, mito_table = load_mito_level(data_dir, qc)
        axon_df = build_axon_table(load_cell_volumes(data_dir), mito_df)

    # panel_width/quant_row_h match figure2_mito_level's own panel size (13.5in / 3 cols,
    # 9in / 2 rows) so titles that already fit there don't overflow into the letter label here.
    panel_width = 4.5
    quant_row_h = 4.5
    img_nrows = 2 if top_view == "all" else 1
    img_ncols = 3 if top_view == "all" else 2

    fig_w = panel_width * 3
    img_col_w = fig_w / img_ncols
    img_row_h = img_col_w * target_h / target_w
    content_h = img_row_h * img_nrows + quant_row_h * 2
    top_margin, bottom_margin = 1.2, 1.1  # inches reserved for suptitle / legend
    if quant == "manual_vs_auto":
        bottom_margin = 2.5   # genotype label under the ticks + a three-line legend
    fig_h = content_h + top_margin + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[img_row_h * img_nrows, quant_row_h * 2], hspace=0.10,
        top=1 - top_margin / fig_h, bottom=bottom_margin / fig_h,
        left=0.05, right=0.99,
    )
    quant_gs = outer[1].subgridspec(2, 3, wspace=0.25, hspace=0.5)

    if top_view == "all":
        img_gs = outer[0].subgridspec(2, 3, wspace=0.05, hspace=0.12)
        build_reconstruction_block(fig, img_gs, screenshot_dir, target_size, calib)
    else:
        img_gs = outer[0].subgridspec(1, 2, wspace=0.05)
        build_single_view_block(fig, img_gs, screenshot_dir, target_size, calib, top_view)

    if quant == "manual_vs_auto":
        build_quant_block_mva(fig, quant_gs, mito_df, manual_animals)
        handles = mva_legend_handles(manual_animals)
        title = None  # no heading on this version
    else:
        build_quant_block(fig, quant_gs, axon_df, mito_df)
        handles = _legend_handles([
            ("mito", "one mitochondrion — violin and box show this distribution"),
            ("axon", "one axon, median of its mitochondria"),
        ])
        title = ("Mitochondria — 3D reconstructions and morphometrics "
                 "(4010 WT vs 4007 $\\mathit{Plp}^{-/y}$)")

    if legend:
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   fontsize=14, bbox_to_anchor=(0.5, 0.25 / fig_h))

    if title:
        fig.suptitle(title, fontsize=20, y=1 - 0.3 / fig_h)
    return fig, mito_table, handles


def save_legend(handles, legend_dir, name, png):
    """The figure legend on its own, for placing by hand next to the legend-free figure."""
    os.makedirs(legend_dir, exist_ok=True)
    fig = plt.figure()
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=14)
    for ext in ["pdf"] + (["png"] if png else []):
        path = os.path.join(legend_dir, f"{name}_legend.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screenshot-dir", default="data",
                        help="dir holding 4010_3d/4007_3d screenshot folders")
    parser.add_argument("--data-dir", default="data",
                        help="dir holding the per-sample morphometrics CSV folders")
    parser.add_argument("--qc", action="store_true", help="use the QC'd mito table")
    parser.add_argument("--top-view", choices=["all"] + VIEWS, default="all",
                        help="'all' = 2x3 (genotype x view) reconstruction block; a single "
                             "view name = 1x2 (WT vs Plp-/y) block showing just that view")
    parser.add_argument("--quant", choices=["fig2", "manual_vs_auto"], default="fig2",
                        help="lower block: fig2 = WT vs Plp-/y automatic only; "
                             "manual_vs_auto = the four-violin manual (IMOD) vs automatic "
                             "panels from figure_manual_vs_auto.py")
    parser.add_argument("--imod-variant", default="z25", choices=("z25", "full"),
                        help="--quant manual_vs_auto only: which IMOD run feeds the manual side")
    parser.add_argument("--manual-animals", default="all", choices=("matched", "all"),
                        help="--quant manual_vs_auto only: matched = 4010/4007 only, "
                             "all = the whole IMOD set (3 animals per genotype)")
    parser.add_argument("--out-dir", default="figures", help="Plots/ is created here")
    parser.add_argument("--png", action="store_true", help="also write a PNG")
    parser.add_argument("--legend-dir", default=None,
                        help="leave the legend off the figure and write it to this dir "
                             "as its own file instead")
    args = parser.parse_args()

    name = "figure2_combined" if args.top_view == "all" else f"figure2_combined_{args.top_view}"
    if args.quant == "manual_vs_auto":
        name += "_manual_vs_auto"
        if args.manual_animals == "all":
            name += "_all_animals"
    if args.qc:
        name += "_qc"

    build_kwargs = dict(quant=args.quant, imod_variant=args.imod_variant,
                        manual_animals=args.manual_animals,
                        legend=args.legend_dir is None)
    fig, mito_table, handles = build_combined_figure(args.screenshot_dir, args.data_dir, args.qc,
                                            args.top_view, **build_kwargs)

    plot_dir = os.path.join(args.out_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    pdf_path = os.path.join(plot_dir, f"{name}.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"mode: {'qc' if args.qc else 'raw'} -- quant panels from {mito_table}")
    print(f"Written: {pdf_path}")

    if args.png:
        plt.close(fig)
        fig, _, _ = build_combined_figure(args.screenshot_dir, args.data_dir, args.qc,
                                       args.top_view, **build_kwargs)
        png_path = os.path.join(plot_dir, f"{name}.png")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Written: {png_path}")

    plt.close(fig)
    if args.legend_dir:
        save_legend(handles, args.legend_dir, name, args.png)


if __name__ == "__main__":
    main()
