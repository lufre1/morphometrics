#!/usr/bin/env python
"""See where the automatic mitochondria segmentation overshoots the hand-traced one.

Opens napari with matched mitochondria from 4010 (WT), side by side:

    left  : raw EM + manual contour (cyan) + automatic contour (magenta)
    right : raw EM + filled agreement map
            grey = both, cyan = manual only, magenta = automatic only

Two sliders: one steps through the selected objects, one through z. Every layer carries
scale=(25, 5, 5) nm so the display is isotropic -- without that the z axis is squashed
5x and you cannot tell a lateral rim from extra slices.

Only 4010 can be shown: 4007's automatic segmentation no longer exists on disk, only its
result CSVs. The three 4010 volumes are already the same shape, so they are voxel-aligned
and no registration is needed.

Usage
-----
    python inspect_overlap_napari.py
    python inspect_overlap_napari.py --n-objects 8 --screenshot overlap.png
    python inspect_overlap_napari.py --no-show          # headless self-test
"""

import argparse
import os

import numpy as np
import pandas as pd
import zarr

from morphometrics_3d import _scan_bboxes

DATA_ROOT = "/home/freckmann15/data/volume-em"
SAMPLE, GENOTYPE = "4010", "WT"

RAW = (f"{DATA_ROOT}/{GENOTYPE}/{SAMPLE}/{SAMPLE}.zarr", "s0")
MANUAL = (f"{DATA_ROOT}/{GENOTYPE}/{SAMPLE}/{SAMPLE}_imod.zarr", "mitos")
# the complete detector output, not the 10-axon _final subset: _final covers only 2 of the
# 5 hand-traced axons, so most manual objects would have no automatic counterpart
AUTO = (f"{DATA_ROOT}/{GENOTYPE}/{SAMPLE}/grid_search_mitos/sd1_bp10_ft07_bt012.zarr", "seg")
MANUAL_CSV = f"{DATA_ROOT}/{GENOTYPE}/{SAMPLE}/{SAMPLE}_morphometrics_imod_full/mito_morphometrics_3d.csv"
AUTO_CSV = f"{DATA_ROOT}/{GENOTYPE}/{SAMPLE}/{SAMPLE}_morphometrics_all_final/mito_morphometrics_3d.csv"

VOXEL_NM = (25.0, 5.0, 5.0)          # z, y, x
PERCENTILES = (10, 30, 50, 70, 90)   # of the automatic/manual volume ratio
GAP_VOXELS = 12                      # between the two panels, in x voxels

BOTH, MANUAL_ONLY, AUTO_ONLY = 1, 2, 3
CYAN, MAGENTA, GREY = "#00e5ff", "#ff2bd6", "#9a9a9a"


def _open(spec):
    path, key = spec
    return zarr.open(path, mode="r")[key]


def manual_bboxes(cache_path, block_shape=(64, 1024, 1024)):
    """{label: [z0, z1, y0, y1, x0, x1]} for the manual mitochondria, cached to CSV.

    The scan is one pass over the whole volume (about a minute), so it is worth keeping.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return {int(r.label): [int(r.z0), int(r.z1), int(r.y0), int(r.y1), int(r.x0), int(r.x1)]
                for r in df.itertuples()}
    print(f"scanning {MANUAL[0]}:{MANUAL[1]} for bounding boxes (cached afterwards) ...")
    bboxes, _counts, _shape = _scan_bboxes(MANUAL[0], MANUAL[1], block_shape)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame([{"label": int(k), **dict(zip("z0 z1 y0 y1 x0 x1".split(), map(int, v)))}
                  for k, v in sorted(bboxes.items())]).to_csv(cache_path, index=False)
    print(f"  wrote {cache_path}")
    return {int(k): list(map(int, v)) for k, v in bboxes.items()}


def crop_slices(bbox, centroid, pad, window, shape):
    """Padded bounding box capped at `window` per axis, centred on the object's centroid.

    Capping matters: these mitochondria span a median of 91 z-slices and the largest far
    more, so padding every crop to the largest would leave small objects as specks in an
    empty frame. A fixed cap keeps the zoom comparable and the stack small.

    Centring on the centroid rather than the bounding-box centre matters just as much.
    They are very elongated (median a/c about 11), and the bbox centre of a long curved
    object can sit almost outside it -- label 58 put only 4k of its 294k voxels inside a
    bbox-centred window. The centroid follows the mass.
    """
    out = []
    for d, (lo, hi) in enumerate(((bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]))):
        size = min(hi - lo + 1 + 2 * pad[d], window[d], shape[d])
        centre = int(round(centroid[d])) if centroid is not None else (lo + hi) // 2
        start = int(np.clip(centre - size // 2, 0, shape[d] - size))
        out.append(slice(start, start + size))
    return tuple(out)


def match_by_overlap(manual_arr, auto_arr, label, bbox, centroid, pad, window, shape):
    """-> (auto_label, slices) for the automatic object sharing most voxels with `label`."""
    sl = crop_slices(bbox, centroid, pad, window, shape)
    man = np.asarray(manual_arr[sl]) == label
    aut = np.asarray(auto_arr[sl])
    hits = aut[man]
    hits = hits[hits != 0]
    if hits.size == 0:
        return None, sl
    vals, counts = np.unique(hits, return_counts=True)
    return int(vals[counts.argmax()]), sl


def select_objects(bboxes, manual_arr, auto_arr, n_objects, pad, window, shape):
    """Match every manual object, then take pairs spanning the volume-ratio range."""
    man_df = pd.read_csv(MANUAL_CSV).set_index("label_id")
    man_vol = man_df.volume_um3
    centroids = man_df[["centroid_z_vx", "centroid_y_vx", "centroid_x_vx"]]
    auto_vol = pd.read_csv(AUTO_CSV).set_index("label_id").volume_um3

    rows = []
    for label, bbox in sorted(bboxes.items()):
        centroid = centroids.loc[label].to_numpy() if label in centroids.index else None
        auto_label, sl = match_by_overlap(manual_arr, auto_arr, label, bbox, centroid,
                                          pad, window, shape)
        if auto_label is None or auto_label not in auto_vol.index or label not in man_vol.index:
            continue
        rows.append(dict(manual_label=label, auto_label=auto_label,
                         manual_um3=float(man_vol[label]), auto_um3=float(auto_vol[auto_label]),
                         ratio=float(auto_vol[auto_label] / man_vol[label]), slices=sl))
    matched = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    print(f"matched {len(matched)} of {len(bboxes)} manual mitochondria; "
          f"median automatic/manual volume ratio {matched.ratio.median():.2f}x")

    pcts = np.linspace(PERCENTILES[0], PERCENTILES[-1], n_objects)
    picks, seen = [], set()
    for p in pcts:
        target = np.percentile(matched.ratio, p)
        for i in (matched.ratio - target).abs().sort_values().index:
            if i not in seen:
                seen.add(i)
                picks.append((p, matched.loc[i]))
                break
    return matched, picks


def build_stacks(picks, raw_arr, manual_arr, auto_arr):
    """-> (raw, manual, auto, agreement) each (n_objects, Z, Y, X), padded to a common shape."""
    crops = []
    for _p, r in picks:
        sl = r.slices
        raw = np.asarray(raw_arr[sl])
        man = (np.asarray(manual_arr[sl]) == r.manual_label)
        aut = (np.asarray(auto_arr[sl]) == r.auto_label)
        agree = np.where(man & aut, BOTH, np.where(man, MANUAL_ONLY,
                                                   np.where(aut, AUTO_ONLY, 0))).astype(np.uint8)
        crops.append((raw, man.astype(np.uint8), aut.astype(np.uint8), agree))

    target = tuple(max(c[0].shape[d] for c in crops) for d in range(3))

    def pad(a):                       # centre each crop in the common box
        before = [(target[d] - a.shape[d]) // 2 for d in range(3)]
        return np.pad(a, [(before[d], target[d] - a.shape[d] - before[d]) for d in range(3)])

    return tuple(np.stack([pad(c[i]) for c in crops]) for i in range(4))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-objects", type=int, default=5)
    ap.add_argument("--pad-xy", type=int, default=20, help="context in y/x voxels (5 nm each)")
    ap.add_argument("--pad-z", type=int, default=4, help="context in z voxels (25 nm each)")
    ap.add_argument("--window", type=int, nargs=3, default=(96, 256, 256),
                    metavar=("Z", "Y", "X"),
                    help="maximum crop size per object; bigger objects are clipped so the "
                         "zoom stays comparable and the stack stays small")
    ap.add_argument("--cache", default="table/4010_manual_mito_bboxes.csv")
    ap.add_argument("--screenshot", default=None, help="save the canvas to this PNG")
    ap.add_argument("--no-show", action="store_true", help="build everything, do not open a window")
    args = ap.parse_args()

    raw_arr, manual_arr, auto_arr = _open(RAW), _open(MANUAL), _open(AUTO)
    shapes = {"raw": raw_arr.shape, "manual": manual_arr.shape, "automatic": auto_arr.shape}
    assert len(set(shapes.values())) == 1, f"volumes are not voxel-aligned: {shapes}"
    shape = raw_arr.shape
    print(f"{SAMPLE} ({GENOTYPE}) volumes aligned at {shape}")

    bboxes = manual_bboxes(args.cache)
    pad = (args.pad_z, args.pad_xy, args.pad_xy)
    matched, picks = select_objects(bboxes, manual_arr, auto_arr,
                                    args.n_objects, pad, tuple(args.window), shape)

    print(f"\nshowing {len(picks)} objects spanning the ratio range:")
    print(f"  {'pct':>4s} {'manual':>7s} {'auto':>7s} {'manual um3':>11s} {'auto um3':>10s} {'ratio':>6s}")
    for p, r in picks:
        print(f"  {p:4.0f} {r.manual_label:7d} {r.auto_label:7d} "
              f"{r.manual_um3:11.4f} {r.auto_um3:10.4f} {r.ratio:6.2f}x")

    raw, man, aut, agree = build_stacks(picks, raw_arr, manual_arr, auto_arr)
    for i, (_p, r) in enumerate(picks):
        present = set(np.unique(agree[i])) - {0}
        assert BOTH in present and AUTO_ONLY in present, (
            f"object {r.manual_label}: agreement map lacks "
            f"{'overlap' if BOTH not in present else 'automatic-only voxels'} ({present})")
    print(f"\ncrop stack {raw.shape} (objects, z, y, x); every object has overlap and an "
          f"automatic-only rim")

    if args.no_show and not args.screenshot:
        return

    import napari
    from napari.utils import DirectLabelColormap

    dx = (raw.shape[3] + GAP_VOXELS) * VOXEL_NM[2]      # offset of the right panel, in nm
    off = (0.0, 0.0, dx)
    sc = VOXEL_NM
    v = napari.Viewer(title=f"{SAMPLE} ({GENOTYPE}) manual vs automatic mitochondria")

    clim = (float(np.percentile(raw, 1)), float(np.percentile(raw, 99)))
    v.add_image(raw, name="raw (contours)", colormap="gray", contrast_limits=clim, scale=sc)
    v.add_image(raw, name="raw (agreement)", colormap="gray", contrast_limits=clim,
                scale=sc, translate=off)

    v.add_labels(man, name="manual", scale=sc, opacity=1.0,
                 colormap=DirectLabelColormap(color_dict={1: CYAN, None: "transparent"}))
    v.add_labels(aut, name="automatic", scale=sc, opacity=1.0,
                 colormap=DirectLabelColormap(color_dict={1: MAGENTA, None: "transparent"}))
    for name in ("manual", "automatic"):
        v.layers[name].contour = 2

    v.add_labels(agree, name="agreement", scale=sc, translate=off, opacity=0.65,
                 colormap=DirectLabelColormap(color_dict={
                     BOTH: GREY, MANUAL_ONLY: CYAN, AUTO_ONLY: MAGENTA, None: "transparent"}))

    # Open on the median-ratio object, not the first: the first is the 10th percentile,
    # where the automatic mask is smaller than the manual one, which is a misleading
    # first impression of a systematic overestimate. And pick the slice where that object
    # is largest -- they are elongated, so the geometric middle is often nearly empty.
    start_obj = len(picks) // 2
    v.dims.set_current_step(0, start_obj)
    v.dims.set_current_step(1, int(man[start_obj].sum(axis=(1, 2)).argmax()))
    v.reset_view()

    if args.screenshot:
        v.screenshot(args.screenshot, canvas_only=True)
        print(f"wrote {args.screenshot}")
    if not args.no_show:
        napari.run()


if __name__ == "__main__":
    main()
