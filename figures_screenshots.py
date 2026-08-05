#!/usr/bin/env python
"""Build a figure with 3D screenshot panels for comparing 4007 (KO) vs 4010 (WT).

Layout options:
  - 2x3: 2 rows (genotypes) × 3 columns (views)
         Row 1: WT, Row 2: KO
         Cols: [Axons, Axons+Mitos, Mitochondria]
  - 3x2: 3 rows (views) × 2 columns (genotypes)
         Cols: [WT, KO]
         Rows: [Axons, Axons+Mitos, Mitochondria]

Usage:
    python figures_screenshots.py          # 2x3 default, PDF only
    python figures_screenshots.py --layout 3x2
    python figures_screenshots.py --png    # also write PNG
    python figures_screenshots.py --layout 3x2 --png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

WT_COLOR = "#1b9e77"
KO_COLOR = "#fc8d62"

SAMPLES = {
    "4010": "WT",
    "4007": r"$\mathit{Plp}^{-/y}$",
}

VIEWS = ["axons", "mitos", "axons_mitos"]

VIEW_LABELS = {
    "axons": "Axons",
    "axons_mitos": "Axons + Mitochondria",
    "mitos": "Mitochondria",
}


# ------------------------------------------------------------
# IMAGE UTILS
# ------------------------------------------------------------

def load_and_pad(image_path, target_shape):
    """Load PNG, scale it to fill target_shape (h, w, c) preserving aspect ratio, pad the rest.

    Source screenshots differ in native pixel size (4007 renders are ~6% larger than 4010's),
    so pasting them 1:1 onto a shared canvas -- the old behaviour -- left 4010 looking smaller
    than 4007 in every panel. Scaling each image up to fill the canvas first removes that
    resolution artifact so panel content is a consistent size across genotypes.
    """
    img = mpimg.imread(image_path)

    target_h, target_w = target_shape[:2]
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pil_img = Image.fromarray((img * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    img = np.asarray(pil_img, dtype=np.float32) / 255.0

    channels = img.shape[-1]
    bg = np.ones((target_h, target_w, channels), dtype=np.float32)
    if channels == 4:
        bg[:, :, 3] = 1.0

    y_pad = (target_h - new_h) // 2
    x_pad = (target_w - new_w) // 2
    bg[y_pad:y_pad + new_h, x_pad:x_pad + new_w, :] = img
    return bg


def compute_target_size(image_paths):
    """Return (max_height, max_width) across all images."""
    max_h, max_w = 0, 0
    for path in image_paths:
        img = mpimg.imread(path)
        max_h = max(max_h, img.shape[0])
        max_w = max(max_w, img.shape[1])
    return (max_h, max_w)


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

def plot_2x3(fig, data_dir, target_size):
    """Layout: 2 rows (genotypes) × 3 columns (views)."""
    axes = []
    
    # Row 1: WT, Row 2: KO
    genotypes = ["4010", "4007"]
    
    for row_idx, (sample_id, genotype_label) in enumerate(SAMPLES.items()):
        for col_idx, view in enumerate(VIEWS):
            ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1)
            axes.append(ax)
            
            img_dir = os.path.join(data_dir, f"{sample_id}_3d")
            filename = f"{sample_id}_{view}_3d.png"
            image_path = os.path.join(img_dir, filename)
            
            img = load_and_pad(image_path, target_size)
            ax.imshow(img)
            ax.axis("off")
            
            if row_idx == 0:
                ax.set_title(VIEW_LABELS[view], fontsize=11, pad=10)
            
            if col_idx == 0:
                ax.text(-0.15, 0.5, genotype_label, transform=ax.transAxes,
                        fontsize=12, va="center", ha="right",
                        rotation=90)
    
    return axes


def plot_3x2(fig, data_dir, target_size):
    """Layout: 3 rows (views) × 2 columns (genotypes)."""
    axes = []
    
    # Cols: WT (4010), KO (4007)
    sample_ids = ["4010", "4007"]
    
    for row_idx, view in enumerate(VIEWS):
        for col_idx, sample_id in enumerate(sample_ids):
            ax = fig.add_subplot(3, 2, row_idx * 2 + col_idx + 1)
            axes.append(ax)
            
            img_dir = os.path.join(data_dir, f"{sample_id}_3d")
            filename = f"{sample_id}_{view}_3d.png"
            image_path = os.path.join(img_dir, filename)
            
            img = load_and_pad(image_path, target_size)
            ax.imshow(img)
            ax.axis("off")
            
            if row_idx == 0:
                ax.set_title(SAMPLES[sample_id], fontsize=12, pad=10)
            
            if col_idx == 0:
                ax.text(-0.05, 0.5, VIEW_LABELS[view], transform=ax.transAxes,
                        fontsize=11, va="center", ha="right", rotation=90)
    
    return axes


def build_figure(data_dir, layout="2x3"):
    """Build the combined screenshot figure."""
    image_dir = os.path.join(data_dir, "4010_3d")
    image_dir_4007 = os.path.join(data_dir, "4007_3d")
    
    # Collect all image paths to compute target size
    image_paths = []
    for sample_id in ["4010", "4007"]:
        img_dir = os.path.join(data_dir, f"{sample_id}_3d")
        for view in VIEWS:
            image_paths.append(os.path.join(img_dir, f"{sample_id}_{view}_3d.png"))
    
    target_size = compute_target_size(image_paths)

    # ponytail: figsize is derived from the screenshot aspect ratio (not hardcoded) so each
    # panel cell matches the image shape -- otherwise imshow letterboxes and leaves an uneven
    # gap between panels. panel_width is the only knob; tweak it to scale the whole figure.
    target_h, target_w = target_size
    panel_width = 3.2
    panel_height = panel_width * target_h / target_w

    if layout == "2x3":
        ncols, nrows = 3, 2
        fig = plt.figure(figsize=(panel_width * ncols, panel_height * nrows))
        plot_2x3(fig, data_dir, target_size)
        fig.suptitle("3D reconstructions — 4010 (WT) vs 4007 (Plp⁻/y)", fontsize=14, y=0.98)
    else:
        ncols, nrows = 2, 3
        fig = plt.figure(figsize=(panel_width * ncols, panel_height * nrows))
        plot_3x2(fig, data_dir, target_size)
        fig.suptitle("3D reconstructions — 4010 (WT) vs 4007 (Plp⁻/y)", fontsize=14, y=0.99)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.3, h_pad=0.2, w_pad=0.2)
    return fig


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="directory with 4007_3d and 4010_3d")
    parser.add_argument("--out-dir", default="figures", help="output directory")
    parser.add_argument("--layout", choices=["2x3", "3x2"], default="2x3",
                        help="layout: 2x3 = genotypes as rows, 3x2 = views as rows")
    parser.add_argument("--png", action="store_true", help="also write PNG version")
    args = parser.parse_args()
    
    image_dir = os.path.join(args.data_dir, "4010_3d")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing {image_dir}")
    
    image_dir_4007 = os.path.join(args.data_dir, "4007_3d")
    if not os.path.isdir(image_dir_4007):
        raise FileNotFoundError(f"Missing {image_dir_4007}")
    
    fig = build_figure(args.data_dir, args.layout)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    pdf_path = os.path.join(args.out_dir, "figure3_screenshots.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {pdf_path}")
    
    if args.png:
        png_path = os.path.join(args.out_dir, "figure3_screenshots.png")
        fig = build_figure(args.data_dir, args.layout)
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Written: {png_path}")


if __name__ == "__main__":
    main()