"""
3D morphometrics for mitochondria (and optionally cells) from volumetric TIFF or Zarr v2 data.

Metric mapping vs. morphometrics_2d.py
---------------------------------------
  area_px          -> volume_vx           (voxel count)
  area_nm2         -> volume_um3          (µm³; nm³ is unwieldy at EM scale)
  perimeter_px     -> OMITTED             (no clean 3D analog; surface area replaces it)
  perimeter_nm     -> surface_um2         (marching-cubes mesh; skip with --no_surface)
  circularity      -> sphericity          (π^(1/3)·(6V)^(2/3)/S; 1 = perfect sphere)
  major/minor axis -> a_um, b_um, c_um    (principal semi-axes from covariance PCA)
  mean/min/max int -> same + std_intensity

Additional 3D metrics not in the 2D script:
  sv_ratio, elongation/flatness/isotropy, euler_char, centroid (vx + µm),
  nearest_neighbor_um, touches_border, cell_id (if cell seg provided),
  skeleton_length_um (opt-in via --skeleton; branches/endpoints are TODO).

Voxel size: supplied in nm on the CLI (consistent with 2D scripts); stored and
reported in µm in all output columns.

Input formats
-------------
  TIFF  — any .tif/.tiff file; keys are ignored.
  Zarr  — a .zarr directory (Zarr v2 DirectoryStore); use --ext .zarr and
          --raw_key / --mito_key / --cell_key (default "s0") to select the
          dataset within the store.

Usage example (TIFF)
--------------------
  python scripts/morphometrics_3d_claude.py \\
      -p  /data/raw/ \\
      -mlpth /data/mito_segs/ \\
      -clpth /data/cell_segs/ \\
      -rp raw -mlp mito -clp cell \\
      -r -o /data/results/

Usage example (Zarr v2)
-----------------------
  python scripts/morphometrics_3d_claude.py \\
      -p  /data/raw.zarr \\
      -mlpth /data/mito.zarr \\
      -e .zarr --raw_key s0 --mito_key s0 \\
      -r -o /data/results/

Outputs
-------
  <output_path>/mito_morphometrics_3d.csv   -- one row per mitochondrion
  <output_path>/cell_summary_3d.csv         -- one row per cell (if -clpth given)
"""

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree
from skimage.measure import euler_number, label as cc_label, marching_cubes
from skimage.morphology import skeletonize
from skimage.segmentation import relabel_sequential
from skimage.transform import resize as sk_resize
from tifffile import imread
from tqdm import tqdm

import synapse.util as util


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_paths(search_dir, ext):
    """Return sorted list of paths matching *ext* under *search_dir*.

    For '.zarr': finds directories ending in '.zarr' (Zarr stores are dirs).
    For all other extensions: delegates to util.get_file_paths.
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    # If the path itself is a Zarr store (any *.zarr / *.ome.zarr directory),
    # return it directly regardless of the ext argument.
    if search_dir.lower().endswith(".zarr") and os.path.isdir(search_dir):
        return [search_dir]
    if ext.lower() == ".zarr":
        candidates = glob(os.path.join(search_dir, "**", "*.zarr"), recursive=True)
        return sorted(p for p in candidates if os.path.isdir(p))
    return util.get_file_paths(search_dir, ext)


def _load_volume(path, key):
    """Load a 3-D volume from a Zarr store (directory) or a TIFF file."""
    if os.path.isdir(path):
        return zarr.open(path, mode='r')[key][:]
    return imread(path)


def _open_lazy(path, key):
    """Return a lazy array view without loading data into memory.

    For Zarr: returns the zarr.Array (no data read).
    For TIFF: attempts tifffile.memmap (uncompressed TIFFs only); falls back to
    a full in-memory load, which will be sliced per block in _scan_bboxes.
    """
    if os.path.isdir(path):
        return zarr.open(path, mode='r')[key]
    import tifffile as _tf
    try:
        return _tf.memmap(path)
    except Exception:
        return imread(path)


def _scan_bboxes(path, key, block_shape=(64, 512, 512)):
    """Scan a label volume in 3-D blocks to build per-ID tight bounding boxes.

    Keeps only one block in memory at a time — memory use is O(block_size).

    Returns
    -------
    bboxes    : dict {label_id -> [z_min, z_max, y_min, y_max, x_min, x_max]}
    vcounts   : dict {label_id -> voxel_count}
    vol_shape : (Z, Y, X)
    """
    arr = _open_lazy(path, key)
    vol_shape = arr.shape
    Z, Y, X = vol_shape
    bz, by, bx = block_shape

    # Quick check: if binary label with relabelling needed, fail early.
    sample = np.asarray(arr[0:min(bz, Z), 0:min(by, Y), 0:min(bx, X)])
    nonzero_ids = set(np.unique(sample).tolist()) - {0}
    if nonzero_ids and max(nonzero_ids) == 1:
        warnings.warn(
            "Binary (0/1) label detected. The lazy pipeline treats the entire foreground "
            "as a single object (ID=1). Pre-compute connected-component instance labels "
            "with skimage.measure.label if you need per-mitochondrion analysis.",
            stacklevel=2,
        )

    bboxes: dict = {}
    vcounts: dict = {}
    total_blocks = (
        ((Z + bz - 1) // bz) * ((Y + by - 1) // by) * ((X + bx - 1) // bx)
    )
    with tqdm(total=total_blocks, desc="  Scanning label volume", leave=False) as pbar:
        for z0 in range(0, Z, bz):
            for y0 in range(0, Y, by):
                for x0 in range(0, X, bx):
                    z1 = min(z0 + bz, Z)
                    y1 = min(y0 + by, Y)
                    x1 = min(x0 + bx, X)
                    chunk = np.asarray(arr[z0:z1, y0:y1, x0:x1])

                    ids = np.unique(chunk)
                    ids = ids[ids != 0]
                    for mid in ids:
                        coords = np.argwhere(chunk == mid)
                        gz = coords[:, 0] + z0
                        gy = coords[:, 1] + y0
                        gx = coords[:, 2] + x0
                        vcounts[mid] = vcounts.get(mid, 0) + len(coords)
                        if mid not in bboxes:
                            bboxes[mid] = [int(gz.min()), int(gz.max()),
                                           int(gy.min()), int(gy.max()),
                                           int(gx.min()), int(gx.max())]
                        else:
                            b = bboxes[mid]
                            b[0] = min(b[0], int(gz.min())); b[1] = max(b[1], int(gz.max()))
                            b[2] = min(b[2], int(gy.min())); b[3] = max(b[3], int(gy.max()))
                            b[4] = min(b[4], int(gx.min())); b[5] = max(b[5], int(gx.max()))
                    pbar.update(1)

    return bboxes, vcounts, vol_shape


def _bb_to_slices(bbox, shape, margin=1):
    """Convert [z0,z1,y0,y1,x0,x1] tight bbox to margin-padded tuple of slices."""
    z0, z1, y0, y1, x0, x1 = bbox
    return (
        slice(max(0, z0 - margin), min(shape[0], z1 + margin + 1)),
        slice(max(0, y0 - margin), min(shape[1], y1 + margin + 1)),
        slice(max(0, x0 - margin), min(shape[2], x1 + margin + 1)),
    )


def _load_subvolume_from_path(path, key, slices):
    """Open a file and read only the region defined by *slices*.

    For Zarr: re-opens the store (safe for use in worker processes).
    For TIFF: uses memmap for uncompressed files, full load otherwise.
    """
    if os.path.isdir(path):
        return np.asarray(zarr.open(path, mode='r')[key][slices])
    import tifffile as _tf
    try:
        arr = _tf.memmap(path)
        return np.array(arr[slices])
    except Exception:
        return imread(path)[slices]


# ---------------------------------------------------------------------------
# Helper functions (ported / adapted from analyze_mito_morph.py)
# ---------------------------------------------------------------------------

def _is_binary_label_image(lbl):
    """Return True if label image contains only values in {0, 1}."""
    if lbl.dtype == bool:
        return True
    u = np.unique(lbl)
    return np.array_equal(u, [0]) or np.array_equal(u, [1]) or np.array_equal(u, [0, 1])


def _ensure_instance_labels(label, connectivity=1):
    """Convert binary mask to CC labels, or re-index non-sequential instance labels."""
    label = np.asarray(label)
    if _is_binary_label_image(label):
        return cc_label(label.astype(bool), connectivity=connectivity)
    if not np.issubdtype(label.dtype, np.integer):
        label = label.astype(np.int64, copy=False)
    label, _, _ = relabel_sequential(label)
    return label


def _bbox_from_coords(coords, shape, margin=1):
    """Bounding-box slices around *coords* with an optional voxel *margin*."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    starts = np.maximum(mins - margin, 0)
    ends = np.minimum(maxs + margin, shape)
    return tuple(slice(s, e) for s, e in zip(starts, ends))


def _surface_area_um2(obj, spacing_zyx_um):
    """Surface area (µm²) of a binary 3-D object via marching cubes.

    Returns NaN when the object is too small for marching cubes (< 2 voxels
    in any dimension of its tight bounding box).
    """
    if any(s < 2 for s in obj.shape):
        return np.nan
    try:
        verts, faces, _, _ = marching_cubes(obj.astype(np.uint8), level=0.5,
                                            spacing=spacing_zyx_um)
    except (ValueError, RuntimeError):
        return np.nan

    def _tri_area(v0, v1, v2):
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    return float(sum(_tri_area(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces))


def _principal_axes_um(scaled_coords):
    """Principal semi-axes (a ≥ b ≥ c, in µm) from covariance PCA.

    Returns ((a, b, c), axis_metrics_dict).
    """
    if len(scaled_coords) < 2:
        nan3 = (np.nan, np.nan, np.nan)
        return nan3, {
            "elongation_a_over_c": np.nan,
            "flatness_b_over_c": np.nan,
            "isotropy_c_over_a": np.nan,
        }
    cov = np.cov(scaled_coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    semi_axes = np.sqrt(np.maximum(np.sort(eigenvalues)[::-1], 0.0))
    a, b, c = semi_axes
    return (a, b, c), {
        "elongation_a_over_c": a / c if c > 0 else np.nan,
        "flatness_b_over_c": b / c if c > 0 else np.nan,
        "isotropy_c_over_a": c / a if a > 0 else np.nan,
    }


def _sphericity(volume_um3, surface_um2):
    """Wadell sphericity: π^(1/3)·(6V)^(2/3) / S.  Range (0, 1], 1 = sphere."""
    if not (np.isfinite(surface_um2) and surface_um2 > 0
            and np.isfinite(volume_um3) and volume_um3 > 0):
        return np.nan
    return float((np.pi ** (1 / 3)) * ((6 * volume_um3) ** (2 / 3)) / surface_um2)


def _skeleton_length_um(obj, spacing_zyx_um):
    """Approximate skeleton length (µm) by counting skeletonised voxels."""
    skel = skeletonize(obj)
    voxel_dist = float(np.linalg.norm(spacing_zyx_um))
    return float(skel.sum()) * voxel_dist


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def morphometrics_3d(
    raw,
    mito_label,
    cell_label=None,
    voxel_size_nm=None,
    relabel=True,
    compute_surface=True,
    compute_skeleton=False,
):
    """Compute 3-D morphometrics for every mitochondrion in *mito_label*.

    Parameters
    ----------
    raw : ndarray, shape (Z, Y, X)
        Raw intensity volume.
    mito_label : ndarray, shape (Z, Y, X)
        Mitochondria segmentation (binary or instance labels).
    cell_label : ndarray, shape (Z, Y, X) or None
        Cell segmentation (binary or instance labels). When provided,
        each mitochondrion is assigned to the cell with the most overlap.
    voxel_size_nm : sequence of 1–3 floats (z [y [x]]) in nm, or None.
        None → all outputs are in pixel / voxel units (µm columns = NaN).
    relabel : bool
        Convert binary masks to CC labels; re-index non-sequential instance
        labels. Applied to both *mito_label* and *cell_label*.
    compute_surface : bool
        Compute surface_um2 via marching cubes (slower).
    compute_skeleton : bool
        Compute skeleton_length_um (slow; branches/endpoints not implemented).

    Returns
    -------
    per_mito_df : pd.DataFrame
    per_cell_df : pd.DataFrame or None
    """
    raw = np.asarray(raw)
    mito_label = np.asarray(mito_label)

    if raw.ndim != 3:
        raise ValueError(f"Expected 3-D raw volume, got shape {raw.shape}")
    if mito_label.shape != raw.shape:
        warnings.warn(
            f"mito_label shape {mito_label.shape} != raw shape {raw.shape}; "
            "resizing with order=0 (nearest-neighbour).",
            stacklevel=2,
        )
        orig_dtype = mito_label.dtype
        mito_label = sk_resize(
            mito_label.astype(np.float32), raw.shape,
            order=0, preserve_range=True, anti_aliasing=False,
        ).astype(orig_dtype)
    if cell_label is not None:
        cell_label = np.asarray(cell_label)
        if cell_label.shape != raw.shape:
            warnings.warn(
                f"cell_label shape {cell_label.shape} != raw shape {raw.shape}; "
                "resizing with order=0 (nearest-neighbour).",
                stacklevel=2,
            )
            orig_dtype = cell_label.dtype
            cell_label = sk_resize(
                cell_label.astype(np.float32), raw.shape,
                order=0, preserve_range=True, anti_aliasing=False,
            ).astype(orig_dtype)

    if relabel:
        mito_label = _ensure_instance_labels(mito_label)
        if cell_label is not None:
            cell_label = _ensure_instance_labels(cell_label)

    # --- voxel size ---
    if voxel_size_nm is not None:
        vs = np.asarray(voxel_size_nm, dtype=float).ravel()
        if vs.size == 1:
            vz_nm = vy_nm = vx_nm = float(vs[0])
        elif vs.size == 2:
            vz_nm, vy_nm = float(vs[0]), float(vs[1])
            vx_nm = vy_nm
        elif vs.size == 3:
            vz_nm, vy_nm, vx_nm = float(vs[0]), float(vs[1]), float(vs[2])
        else:
            raise ValueError(
                f"voxel_size_nm must have 1, 2, or 3 values; got {vs.tolist()}"
            )
        vz_um, vy_um, vx_um = vz_nm * 1e-3, vy_nm * 1e-3, vx_nm * 1e-3
        scale_um = np.array([vz_um, vy_um, vx_um])
        voxel_vol_um3 = float(vz_um * vy_um * vx_um)
        spacing_zyx_um = (vz_um, vy_um, vx_um)
    else:
        scale_um = None
        voxel_vol_um3 = None
        spacing_zyx_um = None

    mito_ids = np.unique(mito_label)
    mito_ids = mito_ids[mito_ids != 0]

    rows = []
    centroids_um = []
    label_list = []

    for mito_id in tqdm(mito_ids, desc="Analysing mitochondria", leave=False):
        mask = mito_label == mito_id
        coords = np.argwhere(mask)
        if coords.size == 0:
            warnings.warn(f"Mito label {mito_id}: empty mask, skipping.")
            continue

        voxel_count = int(mask.sum())
        centroid_vx = coords.mean(axis=0)  # (z, y, x)

        if scale_um is not None:
            scaled_coords = coords.astype(float) * scale_um
            centroid_um = centroid_vx * scale_um
            volume_um3 = float(voxel_count * voxel_vol_um3)
        else:
            scaled_coords = coords.astype(float)
            centroid_um = np.full(3, np.nan)
            volume_um3 = np.nan

        # intensity stats
        pv = raw[mask].astype(float)
        intensity_mean = float(pv.mean())
        intensity_min = float(pv.min())
        intensity_max = float(pv.max())
        intensity_std = float(pv.std())

        # principal axes
        (a_um, b_um, c_um), ax_metrics = _principal_axes_um(scaled_coords)

        # border touch
        touches_border = bool(
            coords[:, 0].min() == 0 or coords[:, 0].max() == mito_label.shape[0] - 1
            or coords[:, 1].min() == 0 or coords[:, 1].max() == mito_label.shape[1] - 1
            or coords[:, 2].min() == 0 or coords[:, 2].max() == mito_label.shape[2] - 1
        )

        # subvolume for Euler number and surface
        bb = _bbox_from_coords(coords, mito_label.shape, margin=1)
        obj = (mito_label[bb] == mito_id)

        try:
            euler_char = int(euler_number(obj, connectivity=3))
        except Exception:
            euler_char = 0

        surface_um2 = np.nan
        sphericity = np.nan
        sv_ratio = np.nan
        if compute_surface and spacing_zyx_um is not None:
            surface_um2 = _surface_area_um2(obj, spacing_zyx_um)
            if np.isfinite(surface_um2) and surface_um2 > 0:
                sv_ratio = float(surface_um2 / volume_um3) if volume_um3 and volume_um3 > 0 else np.nan
                sphericity = _sphericity(volume_um3, surface_um2)

        skel_len = np.nan
        if compute_skeleton:
            skel_len = _skeleton_length_um(obj, spacing_zyx_um) if spacing_zyx_um else np.nan

        # cell assignment
        cell_id = np.nan
        if cell_label is not None:
            cell_values = cell_label[mask]
            cell_ids_in_mito = cell_values[cell_values != 0]
            if cell_ids_in_mito.size > 0:
                vals, counts = np.unique(cell_ids_in_mito, return_counts=True)
                cell_id = int(vals[counts.argmax()])

        row = {
            "label_id": int(mito_id),
            "cell_id": cell_id,
            "volume_vx": voxel_count,
            "volume_um3": volume_um3,
            "surface_um2": surface_um2,
            "sphericity": sphericity,
            "sv_ratio": sv_ratio,
            "a_um": float(a_um),
            "b_um": float(b_um),
            "c_um": float(c_um),
            "elongation_a_over_c": ax_metrics["elongation_a_over_c"],
            "flatness_b_over_c": ax_metrics["flatness_b_over_c"],
            "isotropy_c_over_a": ax_metrics["isotropy_c_over_a"],
            "euler_char": euler_char,
            "touches_border": touches_border,
            "centroid_z_vx": float(centroid_vx[0]),
            "centroid_y_vx": float(centroid_vx[1]),
            "centroid_x_vx": float(centroid_vx[2]),
            "centroid_z_um": float(centroid_um[0]),
            "centroid_y_um": float(centroid_um[1]),
            "centroid_x_um": float(centroid_um[2]),
            "nearest_neighbor_um": np.nan,  # filled below
            "mean_intensity": intensity_mean,
            "min_intensity": intensity_min,
            "max_intensity": intensity_max,
            "std_intensity": intensity_std,
        }
        if compute_skeleton:
            row["skeleton_length_um"] = skel_len

        rows.append(row)
        centroids_um.append(centroid_um)
        label_list.append(int(mito_id))

    per_mito_df = pd.DataFrame(rows)

    # nearest-neighbour distances
    if len(centroids_um) >= 2 and scale_um is not None:
        pts = np.vstack(centroids_um)
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
        per_mito_df["nearest_neighbor_um"] = nn

    if cell_label is None or per_mito_df.empty:
        return per_mito_df, None

    # ------------------------------------------------------------------
    # Per-cell aggregation
    # ------------------------------------------------------------------
    cell_ids_all = np.unique(cell_label)
    cell_ids_all = cell_ids_all[cell_ids_all != 0]

    cell_rows = []
    for cid in tqdm(cell_ids_all, desc="Aggregating per cell", leave=False):
        cell_mask = cell_label == cid
        cell_vx = int(cell_mask.sum())
        if cell_vx == 0:
            continue

        cell_vol_um3 = float(cell_vx * voxel_vol_um3) if voxel_vol_um3 else np.nan

        # cell surface / sphericity
        cell_coords = np.argwhere(cell_mask)
        bb_c = _bbox_from_coords(cell_coords, cell_label.shape, margin=1)
        cell_obj = (cell_label[bb_c] == cid)

        cell_surface_um2 = np.nan
        cell_sphericity = np.nan
        if compute_surface and spacing_zyx_um is not None:
            cell_surface_um2 = _surface_area_um2(cell_obj, spacing_zyx_um)
            cell_sphericity = _sphericity(cell_vol_um3, cell_surface_um2)

        # mito stats for this cell
        mitodf_c = per_mito_df[per_mito_df["cell_id"] == cid]
        mito_count = len(mitodf_c)
        mito_vol_um3 = float(mitodf_c["volume_um3"].sum()) if mito_count > 0 else 0.0
        mito_vol_frac = (mito_vol_um3 / cell_vol_um3) if (np.isfinite(cell_vol_um3) and cell_vol_um3 > 0) else np.nan
        mito_mean_vol_um3 = float(mitodf_c["volume_um3"].mean()) if mito_count > 0 else np.nan
        mito_density = (mito_count / cell_vol_um3) if (np.isfinite(cell_vol_um3) and cell_vol_um3 > 0) else np.nan

        cell_rows.append({
            "label_id": int(cid),
            "volume_vx": cell_vx,
            "volume_um3": cell_vol_um3,
            "surface_um2": cell_surface_um2,
            "sphericity": cell_sphericity,
            "mito_count": mito_count,
            "mito_volume_um3": mito_vol_um3,
            "mito_volume_fraction": mito_vol_frac,
            "mito_mean_volume_um3": mito_mean_vol_um3,
            "mito_density_per_um3": mito_density,
        })

    per_cell_df = pd.DataFrame(cell_rows)
    return per_mito_df, per_cell_df


# ---------------------------------------------------------------------------
# Parallel worker (must be a top-level function to be picklable)
# ---------------------------------------------------------------------------

def _compute_mito_metrics_worker(task):
    """Compute all morphometrics for one mitochondrion from its bounding-box subvolume.

    Designed to run in a worker process: re-opens files from paths so no
    large arrays are sent over IPC.
    """
    (raw_path, raw_key, mito_path, mito_key, cell_path, cell_key,
     mito_id, mito_slices, vol_shape, voxel_size_nm,
     compute_surface, compute_skeleton, cell_scale, mito_scale) = task

    # When mito segmentation is at coarser resolution than raw, build raw-space slices
    # and zoom the mito subvolume up to raw resolution with nearest-neighbour interpolation.
    if mito_scale != (1, 1, 1):
        raw_slices = tuple(
            slice(s.start * sc, min(s.stop * sc, vol_shape[i]))
            for i, (s, sc) in enumerate(zip(mito_slices, mito_scale))
        )
    else:
        raw_slices = mito_slices

    raw_bb  = _load_subvolume_from_path(raw_path,  raw_key,  raw_slices)
    mito_bb = _load_subvolume_from_path(mito_path, mito_key, mito_slices)

    if mito_scale != (1, 1, 1):
        orig_dtype = mito_bb.dtype
        mito_bb = sk_resize(
            mito_bb.astype(np.float32), raw_bb.shape,
            order=0, preserve_range=True, anti_aliasing=False,
        ).astype(orig_dtype)

    cell_bb = None
    if cell_path:
        if cell_scale != (1, 1, 1):
            cell_slices = tuple(
                slice(s.start // sc, (s.stop + sc - 1) // sc)
                for s, sc in zip(raw_slices, cell_scale)
            )
        else:
            cell_slices = raw_slices
        cell_bb = _load_subvolume_from_path(cell_path, cell_key, cell_slices)

    local_mask = mito_bb == mito_id
    local_coords = np.argwhere(local_mask)
    if local_coords.size == 0:
        return None

    offsets = np.array([s.start for s in raw_slices])
    global_coords = local_coords + offsets
    voxel_count = int(local_mask.sum())
    centroid_vx = global_coords.mean(axis=0)

    # --- voxel size ---
    if voxel_size_nm is not None:
        vs = np.asarray(voxel_size_nm, dtype=float).ravel()
        if vs.size == 1:
            scale_um = np.array([vs[0], vs[0], vs[0]]) * 1e-3
        elif vs.size == 2:
            scale_um = np.array([vs[0], vs[1], vs[1]]) * 1e-3
        else:
            scale_um = vs[:3] * 1e-3
        scaled_coords   = global_coords.astype(float) * scale_um
        centroid_um     = centroid_vx * scale_um
        volume_um3      = float(voxel_count * np.prod(scale_um))
        spacing_zyx_um  = tuple(scale_um.tolist())
    else:
        scaled_coords  = global_coords.astype(float)
        centroid_um    = np.full(3, np.nan)
        volume_um3     = np.nan
        spacing_zyx_um = None

    # --- intensity ---
    pv = raw_bb[local_mask].astype(float)
    intensity_mean = float(pv.mean())
    intensity_min  = float(pv.min())
    intensity_max  = float(pv.max())
    intensity_std  = float(pv.std())

    # --- shape metrics ---
    (a_um, b_um, c_um), ax_metrics = _principal_axes_um(scaled_coords)

    touches_border = bool(
        global_coords[:, 0].min() == 0 or global_coords[:, 0].max() == vol_shape[0] - 1 or
        global_coords[:, 1].min() == 0 or global_coords[:, 1].max() == vol_shape[1] - 1 or
        global_coords[:, 2].min() == 0 or global_coords[:, 2].max() == vol_shape[2] - 1
    )

    try:
        euler_char = int(euler_number(local_mask, connectivity=3))
    except Exception:
        euler_char = 0

    surface_um2 = np.nan
    sphericity  = np.nan
    sv_ratio    = np.nan
    if compute_surface and spacing_zyx_um is not None:
        surface_um2 = _surface_area_um2(local_mask, spacing_zyx_um)
        if np.isfinite(surface_um2) and surface_um2 > 0 and np.isfinite(volume_um3) and volume_um3 > 0:
            sv_ratio   = float(surface_um2 / volume_um3)
            sphericity = _sphericity(volume_um3, surface_um2)

    skel_len = np.nan
    if compute_skeleton and spacing_zyx_um is not None:
        skel_len = _skeleton_length_um(local_mask, spacing_zyx_um)

    # --- cell assignment ---
    cell_id = np.nan
    if cell_bb is not None:
        if cell_scale != (1, 1, 1):
            sc_arr = np.array(cell_scale)
            cell_offsets = np.array([cs.start for cs in cell_slices])
            cell_local = (global_coords // sc_arr) - cell_offsets
            for dim in range(3):
                cell_local[:, dim] = np.clip(cell_local[:, dim], 0, cell_bb.shape[dim] - 1)
            cell_values = cell_bb[cell_local[:, 0], cell_local[:, 1], cell_local[:, 2]]
        else:
            cell_values = cell_bb[local_mask]
        nz = cell_values[cell_values != 0]
        if nz.size > 0:
            vals, counts = np.unique(nz, return_counts=True)
            cell_id = int(vals[counts.argmax()])

    row = {
        "label_id":            int(mito_id),
        "cell_id":             cell_id,
        "volume_vx":           voxel_count,
        "volume_um3":          volume_um3,
        "surface_um2":         surface_um2,
        "sphericity":          sphericity,
        "sv_ratio":            sv_ratio,
        "a_um":                float(a_um),
        "b_um":                float(b_um),
        "c_um":                float(c_um),
        "elongation_a_over_c": ax_metrics["elongation_a_over_c"],
        "flatness_b_over_c":   ax_metrics["flatness_b_over_c"],
        "isotropy_c_over_a":   ax_metrics["isotropy_c_over_a"],
        "euler_char":          euler_char,
        "touches_border":      touches_border,
        "centroid_z_vx":       float(centroid_vx[0]),
        "centroid_y_vx":       float(centroid_vx[1]),
        "centroid_x_vx":       float(centroid_vx[2]),
        "centroid_z_um":       float(centroid_um[0]),
        "centroid_y_um":       float(centroid_um[1]),
        "centroid_x_um":       float(centroid_um[2]),
        "nearest_neighbor_um": np.nan,
        "mean_intensity":      intensity_mean,
        "min_intensity":       intensity_min,
        "max_intensity":       intensity_max,
        "std_intensity":       intensity_std,
    }
    if compute_skeleton:
        row["skeleton_length_um"] = skel_len
    return row


def _aggregate_per_cell_lazy(
    per_mito_df, cell_path, cell_key, vol_shape,
    voxel_size_nm, compute_surface, block_shape, cell_scale=(1, 1, 1),
):
    """Build per-cell summary from per-mito results using lazy cell-label loading."""
    cell_bboxes, cell_vcounts, cell_vol_shape = _scan_bboxes(cell_path, cell_key, block_shape)

    voxel_vol_um3  = None
    spacing_zyx_um = None
    if voxel_size_nm is not None:
        vs = np.asarray(voxel_size_nm, dtype=float).ravel()
        if vs.size == 1:
            scale_um = np.array([vs[0], vs[0], vs[0]]) * 1e-3
        elif vs.size == 2:
            scale_um = np.array([vs[0], vs[1], vs[1]]) * 1e-3
        else:
            scale_um = vs[:3] * 1e-3
        # Cell array may be at a coarser resolution than the mito array.
        # Multiply each axis by the corresponding scale factor before computing
        # cell voxel volume and surface spacing.
        cell_scale_arr = np.array(cell_scale, dtype=float)
        cell_scale_um  = scale_um * cell_scale_arr
        voxel_vol_um3  = float(np.prod(cell_scale_um))
        spacing_zyx_um = tuple(cell_scale_um.tolist())

    cell_rows = []
    for cid, vcount in tqdm(cell_vcounts.items(), desc="  Per-cell aggregation", leave=False):
        cell_vol_um3 = float(vcount * voxel_vol_um3) if voxel_vol_um3 else np.nan

        cell_surface_um2 = np.nan
        cell_sphericity  = np.nan
        if compute_surface and spacing_zyx_um is not None:
            slices   = _bb_to_slices(cell_bboxes[cid], cell_vol_shape, margin=1)
            cell_bb  = _load_subvolume_from_path(cell_path, cell_key, slices)
            cell_obj = cell_bb == cid
            cell_surface_um2 = _surface_area_um2(cell_obj, spacing_zyx_um)
            cell_sphericity  = _sphericity(cell_vol_um3, cell_surface_um2)

        mitodf_c      = per_mito_df[per_mito_df["cell_id"] == cid]
        mito_count    = len(mitodf_c)
        mito_vol_um3  = float(mitodf_c["volume_um3"].sum()) if mito_count > 0 else 0.0
        mito_vol_frac = (
            mito_vol_um3 / cell_vol_um3
            if np.isfinite(cell_vol_um3) and cell_vol_um3 > 0 else np.nan
        )
        mito_mean_vol = float(mitodf_c["volume_um3"].mean()) if mito_count > 0 else np.nan
        mito_density  = (
            mito_count / cell_vol_um3
            if np.isfinite(cell_vol_um3) and cell_vol_um3 > 0 else np.nan
        )

        cell_rows.append({
            "label_id":             int(cid),
            "volume_vx":            vcount,
            "volume_um3":           cell_vol_um3,
            "surface_um2":          cell_surface_um2,
            "sphericity":           cell_sphericity,
            "mito_count":           mito_count,
            "mito_volume_um3":      mito_vol_um3,
            "mito_volume_fraction": mito_vol_frac,
            "mito_mean_volume_um3": mito_mean_vol,
            "mito_density_per_um3": mito_density,
        })

    return pd.DataFrame(cell_rows)


# ---------------------------------------------------------------------------
# QC + cell-level summary
# ---------------------------------------------------------------------------

def _qc_filter_df(mitos, cells,
                  exclude_border=True, min_intensity=1.0,
                  max_elongation=20.0, max_cell_mito_vol_frac=1.0):
    """Apply QC filters to mito and cell DataFrames."""
    n_before = len(mitos)
    if exclude_border and "touches_border" in mitos.columns:
        mitos = mitos[~mitos["touches_border"]].copy()
    if "mean_intensity" in mitos.columns:
        mitos = mitos[mitos["mean_intensity"] >= min_intensity].copy()
    if "elongation_a_over_c" in mitos.columns:
        mitos = mitos[mitos["elongation_a_over_c"] < max_elongation].copy()
    print(f"  QC mitos: {n_before} → {len(mitos)} "
          f"({n_before - len(mitos)} removed: border / low-intensity / elongation ≥ {max_elongation})")

    if cells is None or cells.empty:
        return mitos, cells

    assigned = mitos[mitos["cell_id"].notna()].copy()
    if assigned.empty:
        return mitos, cells

    cell_agg = (
        assigned.groupby("cell_id")
        .agg(mito_count=("label_id", "count"),
             mito_volume_um3=("volume_um3", "sum"))
        .reset_index()
        .rename(columns={"cell_id": "label_id"})
    )
    stale = [c for c in ["mito_count", "mito_volume_um3", "mito_volume_fraction",
                          "mito_mean_volume_um3", "mito_density_per_um3"]
             if c in cells.columns]
    cells = cells.drop(columns=stale).merge(cell_agg, on="label_id", how="left")
    cells["mito_count"] = cells["mito_count"].fillna(0).astype(int)
    cells["mito_volume_um3"] = cells["mito_volume_um3"].fillna(0.0)
    if "volume_um3" in cells.columns:
        cells["mito_volume_fraction"] = cells["mito_volume_um3"] / cells["volume_um3"]

    n_cell_before = len(cells)
    cells = cells[
        (cells["mito_count"] > 0) &
        (cells["mito_volume_fraction"] <= max_cell_mito_vol_frac)
    ].copy()
    print(f"  QC cells:  {n_cell_before} → {len(cells)} "
          f"({n_cell_before - len(cells)} removed: no mitos or vol_frac > {max_cell_mito_vol_frac})")
    return mitos, cells


def _compute_cell_level_summary_df(merged, cells):
    """Build per-cell mito summary from in-memory DataFrames."""
    keep = [c for c in ["label_id", "volume_um3", "mito_count"] if c in cells.columns]
    opt  = [c for c in ["mito_volume_um3", "mito_volume_fraction"] if c in cells.columns]
    cell_metrics = cells[keep + opt].copy().rename(columns={
        "label_id": "cell_id", "volume_um3": "cell_volume_um3",
    })
    if "mito_count" in cell_metrics.columns and "cell_volume_um3" in cell_metrics.columns:
        cell_metrics["mito_count_density"] = (
            cell_metrics["mito_count"] / cell_metrics["cell_volume_um3"]
        )
    if "mito_volume_um3" in cell_metrics.columns:
        cell_metrics["mito_volume_density"] = (
            cell_metrics["mito_volume_um3"] / cell_metrics["cell_volume_um3"]
        )

    agg_spec = [
        ("mito_volume_um3",              [("mean_mito_volume_um3",   "mean"),
                                           ("median_mito_volume_um3", "median"),
                                           ("std_mito_volume_um3",    "std")]),
        ("mito_volume_fraction_of_cell", [("mean_single_mito_fraction_of_cell", "mean")]),
        ("sphericity",                   [("mean_mito_sphericity",   "mean"),
                                           ("median_mito_sphericity", "median")]),
        ("sv_ratio",                     [("mean_mito_sv_ratio",   "mean"),
                                           ("median_mito_sv_ratio", "median")]),
        ("elongation_a_over_c",          [("mean_mito_elongation",   "mean"),
                                           ("median_mito_elongation", "median")]),
        ("flatness_b_over_c",            [("mean_mito_flatness",   "mean"),
                                           ("median_mito_flatness", "median")]),
    ]
    agg_dict = {}
    for col, pairs in agg_spec:
        if col in merged.columns:
            for name, func in pairs:
                agg_dict[name] = (col, func)

    if agg_dict:
        mito_agg = merged.groupby("cell_id").agg(**agg_dict).reset_index()
        full = cell_metrics.merge(mito_agg, on="cell_id", how="inner")
    else:
        full = cell_metrics.copy()

    for col in ["cell_volume_um3", "mito_count", "mito_count_density",
                "mito_volume_um3", "mito_volume_density",
                "mean_mito_volume_um3", "median_mito_volume_um3"]:
        if col in full.columns:
            full[f"log10_{col}"] = np.log10(full[col] + 1e-9)

    return full


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_voxel_size(vs_list):
    """Return None or a tuple of floats from a CLI nargs='+' float list."""
    return tuple(vs_list) if vs_list else None


def main(args):
    ext = args.ext if args.ext is not None else ".tif"
    block_shape = tuple(args.block_shape)
    n_workers = args.n_workers if args.n_workers > 0 else os.cpu_count()

    # --- collect file paths ---
    paths = _find_paths(args.path, ext)
    raw_paths = [p for p in paths if args.raw_pattern in p] if args.raw_pattern else paths

    mito_label_paths = []
    if args.mito_label_path is not None:
        mp = _find_paths(args.mito_label_path, ext)
        mito_label_paths = [p for p in mp if args.mito_label_pattern in p] if args.mito_label_pattern else mp

    cell_label_paths = []
    if args.cell_label_path is not None:
        cp = _find_paths(args.cell_label_path, ext)
        cell_label_paths = [p for p in cp if args.cell_label_pattern in p] if args.cell_label_pattern else cp

    raw_paths = sorted(raw_paths)
    mito_label_paths = sorted(mito_label_paths)
    cell_label_paths = sorted(cell_label_paths)

    if not raw_paths:
        raise FileNotFoundError(f"No raw files found under {args.path!r}")
    if not mito_label_paths:
        raise FileNotFoundError(f"No mito label files found under {args.mito_label_path!r}")

    assert len(raw_paths) == len(mito_label_paths), (
        f"Unequal raw ({len(raw_paths)}) and mito label ({len(mito_label_paths)}) file counts."
    )
    if cell_label_paths:
        assert len(raw_paths) == len(cell_label_paths), (
            f"Unequal raw ({len(raw_paths)}) and cell label ({len(cell_label_paths)}) file counts."
        )

    voxel_size_nm = _resolve_voxel_size(args.voxel_size)
    compute_surface = not args.no_surface

    all_mito_rows = []
    all_cell_rows = []

    for i, (raw_path, mito_path) in enumerate(
        tqdm(list(zip(raw_paths, mito_label_paths)), desc="Files")
    ):
        cell_path = cell_label_paths[i] if cell_label_paths else None

        if args.verbose:
            print(f"raw:  {raw_path}")
            print(f"mito: {mito_path}")
            if cell_path:
                print(f"cell: {cell_path}")

        # Step 1: scan label volume in blocks — O(block_size) peak memory.
        print(f"  Scanning bounding boxes in {os.path.basename(mito_path)} ...")
        bboxes, _vcounts, vol_shape = _scan_bboxes(mito_path, args.mito_key, block_shape)
        mito_ids = sorted(bboxes.keys())
        print(f"  Found {len(mito_ids)} objects.")

        # Step 2: detect mito→raw scale (mito segmentation may be at coarser resolution).
        raw_arr = _open_lazy(raw_path, args.raw_key)
        raw_vol_shape = raw_arr.shape
        del raw_arr
        mito_scale = tuple(max(1, round(raw_vol_shape[i] / vol_shape[i])) for i in range(3))
        if mito_scale != (1, 1, 1):
            print(f"  Mito→raw scale: {mito_scale}  "
                  f"(mito {vol_shape} / raw {raw_vol_shape})")

        # Step 2b: compute scale factor between raw and cell arrays (may differ).
        cell_scale = (1, 1, 1)
        if cell_path is not None:
            cell_arr = _open_lazy(cell_path, args.cell_key)
            cs = tuple(max(1, round(raw_vol_shape[i] / cell_arr.shape[i])) for i in range(3))
            if cs != (1, 1, 1):
                print(f"  Raw→cell scale: {cs}  "
                      f"(raw {raw_vol_shape} / cell {cell_arr.shape})")
            cell_scale = cs

        # Step 3: build per-object tasks (only paths + slices are serialised).
        tasks = [
            (raw_path,  args.raw_key,
             mito_path, args.mito_key,
             cell_path, args.cell_key,
             mito_id,
             _bb_to_slices(bboxes[mito_id], vol_shape, margin=1),  # mito-space slices
             raw_vol_shape, voxel_size_nm, compute_surface, args.skeleton,
             cell_scale, mito_scale)
            for mito_id in mito_ids
        ]

        # Step 4: compute metrics in parallel — each worker loads only its subvolume.
        chunksize = max(1, len(tasks) // (n_workers * 4))
        rows = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for row in tqdm(
                pool.map(_compute_mito_metrics_worker, tasks, chunksize=chunksize),
                total=len(tasks), desc="  Computing metrics", leave=False,
            ):
                if row is not None:
                    rows.append(row)

        per_mito_df = pd.DataFrame(rows)

        # Step 5: nearest-neighbour distances (in main process; centroids are tiny).
        if len(rows) >= 2:
            pts = per_mito_df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].values
            if not np.any(np.isnan(pts)):
                tree = cKDTree(pts)
                dists, _ = tree.query(pts, k=2)
                per_mito_df["nearest_neighbor_um"] = dists[:, 1]

        # Step 6: per-cell aggregation (lazy, no full cell array in memory).
        cell_df = None
        if cell_path is not None and not per_mito_df.empty:
            print("  Aggregating per-cell stats ...")
            cell_df = _aggregate_per_cell_lazy(
                per_mito_df, cell_path, args.cell_key,
                vol_shape, voxel_size_nm, compute_surface, block_shape,
                cell_scale=cell_scale,
            )

        per_mito_df.insert(0, "mito_file", os.path.basename(mito_path))
        if cell_path:
            per_mito_df.insert(0, "cell_file", os.path.basename(cell_path))
        per_mito_df.insert(0, "raw_file", os.path.basename(raw_path))
        all_mito_rows.append(per_mito_df)

        if cell_df is not None and not cell_df.empty:
            cell_df.insert(0, "mito_file", os.path.basename(mito_path))
            cell_df.insert(0, "cell_file", os.path.basename(cell_path) if cell_path else None)
            cell_df.insert(0, "raw_file", os.path.basename(raw_path))
            all_cell_rows.append(cell_df)

    # --- write outputs ---
    out_dir = Path(args.output_path) if args.output_path else Path(args.path) / "morphometrics_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    mito_out = out_dir / "mito_morphometrics_3d.csv"
    out_mito_df = pd.concat(all_mito_rows, ignore_index=True) if all_mito_rows else pd.DataFrame()
    out_mito_df.to_csv(mito_out, index=False)
    print(f"Wrote: {mito_out}  ({len(out_mito_df)} rows)")

    out_cell_df = pd.DataFrame()
    if all_cell_rows:
        cell_out = out_dir / "cell_summary_3d.csv"
        out_cell_df = pd.concat(all_cell_rows, ignore_index=True)
        out_cell_df.to_csv(cell_out, index=False)
        print(f"Wrote: {cell_out}  ({len(out_cell_df)} rows)")

    # --- QC + cell-level summary ---
    if not out_cell_df.empty and not out_mito_df.empty:
        print("\nRunning QC and cell-level summary...")
        qc_mito_df, qc_cell_df = _qc_filter_df(
            out_mito_df.copy(), out_cell_df.copy(),
            exclude_border=not args.no_qc_border,
            min_intensity=args.qc_min_intensity,
            max_elongation=args.qc_max_elongation,
            max_cell_mito_vol_frac=args.qc_max_vol_frac,
        )
        if not qc_mito_df.empty and qc_cell_df is not None and not qc_cell_df.empty:
            valid_cell_ids = set(qc_cell_df["label_id"].values)
            qc_mito_assigned = qc_mito_df[qc_mito_df["cell_id"].isin(valid_cell_ids)].copy()

            merged_df = qc_mito_assigned.merge(
                qc_cell_df[["label_id", "volume_um3"]].rename(
                    columns={"label_id": "cell_id", "volume_um3": "cell_volume_um3"}
                ),
                on="cell_id", how="inner",
            )
            if "volume_um3" in merged_df.columns:
                merged_df = merged_df.rename(columns={"volume_um3": "mito_volume_um3"})
            if "mito_volume_um3" in merged_df.columns and "cell_volume_um3" in merged_df.columns:
                merged_df["mito_volume_fraction_of_cell"] = (
                    merged_df["mito_volume_um3"] / merged_df["cell_volume_um3"]
                )

            qc_mito_out = out_dir / "mito_morphometrics_3d_qc.csv"
            qc_mito_df.to_csv(qc_mito_out, index=False)
            print(f"Wrote: {qc_mito_out}  ({len(qc_mito_df)} rows)")

            cell_summary_df = _compute_cell_level_summary_df(merged_df, qc_cell_df)
            summary_out = out_dir / "cell_level_mito_summary.csv"
            cell_summary_df.to_csv(summary_out, index=False)
            print(f"Wrote: {summary_out}  ({len(cell_summary_df)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="3D morphometrics for mitochondria (and cells) from TIFF or Zarr v2 volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--path", "-p", type=str, required=True,
                    help="Directory or file containing raw TIFF volume(s).")
    ap.add_argument("--mito_label_path", "-mlpth", type=str, default=None,
                    help="Directory or file containing mitochondria segmentation TIFF(s).")
    ap.add_argument("--cell_label_path", "-clpth", type=str, default=None,
                    help="Directory or file containing cell segmentation TIFF(s) (optional).")
    ap.add_argument("--ext", "-e", type=str, default=None,
                    help="File extension filter (e.g. '.tif').")
    ap.add_argument("--output_path", "-o", type=str, default=None,
                    help="Output directory (default: <path>/morphometrics_3d/).")
    ap.add_argument("--raw_pattern", "-rp", type=str, default=None,
                    help="Substring filter for raw file paths.")
    ap.add_argument("--mito_label_pattern", "-mlp", type=str, default=None,
                    help="Substring filter for mito label file paths.")
    ap.add_argument("--cell_label_pattern", "-clp", type=str, default=None,
                    help="Substring filter for cell label file paths.")
    ap.add_argument("--raw_key", type=str, default="s0",
                    help="Dataset key inside raw Zarr store (ignored for TIFF).")
    ap.add_argument("--mito_key", type=str, default="s0",
                    help="Dataset key inside mito label Zarr store (ignored for TIFF).")
    ap.add_argument("--cell_key", type=str, default="s0",
                    help="Dataset key inside cell label Zarr store (ignored for TIFF).")
    ap.add_argument("--voxel_size", "-vs", type=float, nargs="+", default=[25.0, 5.0, 5.0],
                    help="Voxel size in nm: 1 value (isotropic), 2 (z yx), or 3 (z y x). Default: 25 5 5.")
    ap.add_argument("--n_workers", "-nw", type=int, default=0,
                    help="Parallel worker processes (0 = all CPUs).")
    ap.add_argument("--block_shape", type=int, nargs=3, default=[64, 512, 512],
                    help="Z Y X block size for scanning the label volume.")
    ap.add_argument("--relabel", "-r", action="store_true",
                    help="Relabel non-sequential instance labels. Binary inputs require "
                         "pre-computed connected-component labels for large volumes.")
    ap.add_argument("--no_surface", action="store_true",
                    help="Skip marching-cubes surface area (faster).")
    ap.add_argument("--skeleton", action="store_true",
                    help="Compute skeleton length (slow; branches/endpoints not yet implemented).")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no_qc_border", action="store_true",
                    help="Do not exclude mitos touching the volume boundary.")
    ap.add_argument("--qc_min_intensity", type=float, default=1.0,
                    help="Min mean_intensity for mitos (default: 1.0; removes outside-FOV objects).")
    ap.add_argument("--qc_max_elongation", type=float, default=20.0,
                    help="Max elongation_a_over_c for mitos (default: 20; removes merge artifacts).")
    ap.add_argument("--qc_max_vol_frac", type=float, default=1.0,
                    help="Max per-cell mito volume fraction (default: 1.0).")
    args = ap.parse_args()
    main(args)
