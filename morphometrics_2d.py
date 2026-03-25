import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from glob import glob
from tifffile import imread
import numpy as np
import argparse
from skimage.measure import label as cc_label
from skimage.measure import regionprops, perimeter, perimeter_crofton
from skimage.segmentation import relabel_sequential


def get_file_paths(path, ext=".h5", reverse=False):
    ext = ext if ext.startswith(".") else f".{ext}"
    ext_l = ext.lower()

    # If `path` is a file, just check its extension (case-insensitive)
    if os.path.isfile(path):
        return [path] if path.lower().endswith(ext_l) else []

    # Otherwise search recursively and filter case-insensitively
    candidates = glob(os.path.join(path, "**", "*"), recursive=True)
    paths = [p for p in candidates if os.path.isfile(p) and p.lower().endswith(ext_l)]
    return sorted(paths, reverse=reverse)


def mito_summary_per_cell(cell_label, mito_label, relabel_cells=True):
    """
    Returns a dataframe with one row per cell label_id:
      - mito_pixel_count: number of pixels inside the cell that belong to any mito (mito_label > 0)
      - mito_fg_ratio: mito_pixel_count / cell_area_px
      - mito_amount: number of mito instances overlapping the cell (unique mito ids > 0 within the cell)
      - mito_total_pixels_all_instances: sum of pixels for all mito instances in the cell
        (same as mito_pixel_count if mito_label is an instance label image)
    """
    cell_label = np.asarray(cell_label)
    mito_label = np.asarray(mito_label)

    if relabel_cells:
        if _is_binary_label_image(cell_label):
            cell_label = cc_label(cell_label.astype(bool), connectivity=1)
        else:
            if not np.issubdtype(cell_label.dtype, np.integer):
                cell_label = cell_label.astype(np.int64, copy=False)
            cell_label, _, _ = relabel_sequential(cell_label)

    # Treat mito as foreground if it's binary; if instance-labeled, >0 is fg anyway
    mito_fg = (mito_label > 0)

    rows = []
    cell_ids = np.unique(cell_label)
    cell_ids = cell_ids[cell_ids != 0]

    for cid in cell_ids:
        cell_mask = (cell_label == cid)
        cell_area_px = int(cell_mask.sum())
        if cell_area_px == 0:
            continue

        mito_in_cell = mito_label[cell_mask]
        mito_fg_px = int((mito_in_cell > 0).sum())
        mito_fg_ratio = mito_fg_px / float(cell_area_px)

        mito_ids = np.unique(mito_in_cell)
        mito_ids = mito_ids[mito_ids != 0]
        mito_amount = int(len(mito_ids))

        # total mito pixels across all mito instances in the cell
        # (for instance labels, this equals mito_fg_px; kept as explicit column)
        mito_total_pixels = mito_fg_px

        rows.append({
            "label_id": int(cid),
            "cell_area_px": cell_area_px,
            "mito_fg_ratio": mito_fg_ratio,
            "mito_amount": mito_amount,
            "mito_pixel_count": mito_fg_px,
            "mito_total_pixels_all_instances": mito_total_pixels,
        })

    return pd.DataFrame(rows)


def _is_binary_label_image(lbl):
    if lbl.dtype == bool:
        return True
    u = np.unique(lbl)
    # allow {0}, {1}, {0,1}
    return np.array_equal(u, [0]) or np.array_equal(u, [1]) or np.array_equal(u, [0, 1])


def morphometrics(raw, label, voxel_size=None, use_crofton=True, relabel=True):
    raw = np.asarray(raw)
    label = np.asarray(label)
    if relabel:
        # ensure instance labels
        if _is_binary_label_image(label):
            label = cc_label(label.astype(bool), connectivity=1)
        else:
            # ensure ints and sequential ids (optional but convenient)
            if not np.issubdtype(label.dtype, np.integer):
                label = label.astype(np.int64, copy=False)
            label, _, _ = relabel_sequential(label)
    voxel_size = np.asarray(voxel_size, dtype=float).ravel()
    if voxel_size.size == 1:
        s = float(voxel_size[0])
        voxel_size = (s, s)
    elif voxel_size.size == 2:
        voxel_size = (voxel_size[0], voxel_size[1])
    else:
        raise ValueError(f"Voxel size must have 1 or 2 values for 2D, got {voxel_size.tolist()}")
    orig_voxel_size = voxel_size
    if voxel_size is None:
        voxel_size = (1.0, 1.0)  # (y_nm, x_nm)
    y_nm, x_nm = map(float, voxel_size)
    px_area_nm2 = y_nm * x_nm
    px_size_nm = (y_nm + x_nm) / 2.0  # isotropic -> effectively y_nm == x_nm

    rows = []
    for r in regionprops(label, intensity_image=raw):
        lab = r.label
        area_px = r.area
        area_nm2 = area_px * px_area_nm2

        mask = (r.image > 0)  # binary mask in the region's bounding box
        perim_px = perimeter_crofton(mask) if use_crofton else perimeter(mask)
        perim_nm = perim_px * px_size_nm

        circ = np.nan
        if perim_nm > 0:
            circ = 4.0 * np.pi * area_nm2 / (perim_nm ** 2)  # C=4πA/P² 

        rows.append({
            "label_id": lab,
            "area_px": area_px,
            "area_nm2": area_nm2 if orig_voxel_size is not None else None,
            "perimeter_px": perim_px,
            "perimeter_nm": perim_nm if orig_voxel_size is not None else None,
            "circularity": circ,
            "mean_intensity": r.mean_intensity,
            "min_intensity": r.min_intensity,
            "max_intensity": r.max_intensity,
        })

    return pd.DataFrame(rows)


def main(args):
    # load paths
    paths = get_file_paths(args.path, args.ext)

    if args.raw_pattern is not None:
        raw_paths = [path for path in paths if args.raw_pattern in path]
    if args.mito_label_path is not None:
        paths = get_file_paths(args.mito_label_path, args.ext)
    if args.mito_label_pattern is not None:
        mito_label_paths = [path for path in paths if args.mito_label_pattern in path]

    label_paths = None
    if args.label_path is not None:
        paths = get_file_paths(args.label_path, args.ext)
        if args.label_pattern is not None:
            paths = [p for p in paths if args.label_pattern in p]
        label_paths = sorted(paths)
        assert len(mito_label_paths) == len(label_paths), (
            f"Expect equal number of cell label and mito label paths, got {len(label_paths)} and {len(mito_label_paths)}"
        )

    raw_paths.sort()
    label_paths.sort()
    if mito_label_paths is not None:
        mito_label_paths.sort()

    assert len(raw_paths) == len(label_paths), (
        f"Expect equal number of raw and label paths, got {len(raw_paths)} and {len(label_paths)}"
    )
    if args.voxel_size is None:
        print("Warning: voxel size not specified")

    all_rows = []
    for i, (raw_path, label_path) in enumerate(tqdm(list(zip(raw_paths, label_paths)), total=len(raw_paths), desc="Computing morphometrics")):
        raw = imread(raw_path)
        cell_label = imread(label_path)

        df = morphometrics(
            raw,
            cell_label,
            voxel_size=args.voxel_size,
            use_crofton=True,
            relabel=args.relabel,
        )

        # ---- add mito-derived cell summary ----
        if mito_label_paths is not None:
            mito_label = imread(mito_label_paths[i])
            mito_df = mito_summary_per_cell(cell_label, mito_label, relabel_cells=args.relabel)
            # merge on label_id (cell id)
            df = df.merge(mito_df.drop(columns=["cell_area_px"]), on="label_id", how="left")
        df.insert(0, "mito_file", str(os.path.basename(mito_label_paths[i])) if mito_label_paths is not None else None)
        df.insert(0, "label_file", str(os.path.basename(label_path)))
        df.insert(0, "raw_file", str(os.path.basename(raw_path)))
        all_rows.append(df)

    out_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    out = args.output_path
    if out is None:
        out = Path(args.path) / "mito_morphometrics.csv" if args.label_path is None else Path(args.label_path) / "cell_morphometrics_summary.csv"
    else:
        out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print("Wrote:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", "-p", type=str, required=True)
    ap.add_argument("--label_path", "-lpth", type=str, default=None)
    ap.add_argument("--mito_label_path", "-mlpth", type=str, default=None)
    ap.add_argument("--mito_label_pattern", "-mlp", type=str, default=None)
    ap.add_argument("--ext", "-e", type=str, default=None)
    ap.add_argument("--output_path", "-o", type=str, default=None)
    ap.add_argument("--raw_pattern", "-rp", type=str, default=None)
    ap.add_argument("--label_pattern", "-lp", type=str, default=None)
    ap.add_argument("--voxel_size", "-vs", type=float, nargs="+", default=None)
    ap.add_argument("--relabel", "-r", action="store_true", help="relabel instance labels")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    main(args)