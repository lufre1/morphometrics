"""Aggregate the per-animal IMOD morphometrics runs into a cross-animal overview.

Pure aggregation over the verified CSVs produced by run_morphometrics_imod.sh --
nothing is recomputed from the label volumes, so this runs in seconds.

Writes into DATA_ROOT:
  mito_morphometrics_overview.csv   one row per animal (6)
  mito_morphometrics_per_axon.csv   long format, animal x axon x variant x object_set (120)

and, for the two animals that also have an earlier automatically segmented run:
  mito_morphometrics_auto_comparison.csv    animal x pipeline, as published (4)
  mito_morphometrics_auto_per_axon.csv      per axon per pipeline
  mito_morphometrics_auto_matched_mitos.csv manual mitos matched to their nearest automatic one

Run inside `mamba activate synapse`.
"""
import os

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

DATA_ROOT = "/home/freckmann15/data/volume-em"

# animal -> (genotype, source .mod model, raw EM available locally)
ANIMALS = {
    "4005": ("KO", "2019-06-09_PLP_4005.mod", False),
    "4007": ("KO", "2019-08-01_PLP-paper_4007.mod", True),
    "4015": ("KO", "4015_2019-05-09_model_mitos.mod", False),
    "4009": ("WT", "2019-06-12_wt-4009.mod", False),
    "4010": ("WT", "2019-07-31_paper-4010-wt.mod", True),
    "4016": ("WT", "4016_2019-06-11_model_mitos.mod", False),
}
VARIANTS = ("full", "z25")
SETS = {"all": "mito_morphometrics_3d.csv", "qc": "mito_morphometrics_3d_qc.csv"}

# Animals that also have an earlier automatic run in <id>_morphometrics_final/:
# 10 hand-picked axons over a 25 um z-window. animal -> (genotype, nominal z window)
AUTO = {"4007": ("KO", (347, 1347)), "4010": ("WT", (252, 1252))}
MATCH_TOL_UM = 0.5
METRICS = ("n_mito", "mito_volume_um3_total", "mito_volume_fraction_pooled",
           "mito_count_per_100um3", "median_mito_volume_um3", "median_sphericity",
           "median_elongation")


def summarise(mitos, axon_volume_um3):
    """Aggregate a set of per-mito rows against a total axon volume."""
    n = len(mitos)
    total = float(mitos.volume_um3.sum())
    return {
        "n_mito": n,
        "mito_volume_um3_total": total,
        "mito_volume_fraction_pooled": total / axon_volume_um3,
        "mito_count_per_100um3": 100.0 * n / axon_volume_um3,
        "median_mito_volume_um3": float(mitos.volume_um3.median()) if n else np.nan,
        "median_sphericity": float(mitos.sphericity.median()) if n else np.nan,
        "median_elongation": float(mitos.elongation_a_over_c.median()) if n else np.nan,
    }


overview, per_axon = [], []

for sid, (grp, model, has_raw) in ANIMALS.items():
    base = f"{DATA_ROOT}/{grp}/{sid}"
    g = zarr.open(f"{base}/{sid}_imod.zarr", "r")
    vz, vy, vx = g.attrs["voxel_size"]                     # um, z y x
    n_z = g["mitos"].shape[0]
    objs = pd.read_csv(f"{base}/{sid}_imod_objects.csv")
    axon_names = dict(zip(objs[objs.key == "axons"].label, objs[objs.key == "axons"].name))

    # axon volumes come from the full run; identical in z25 (which is why its fractions break)
    cells = pd.read_csv(f"{base}/{sid}_morphometrics_imod_full/cell_summary_3d.csv")
    axon_vol = dict(zip(cells.label_id, cells.volume_um3))
    total_axon_vol = float(cells.volume_um3.sum())

    row = {
        "animal": sid, "genotype": grp, "source_model": model, "has_raw_em": has_raw,
        "voxel_z_nm": vz * 1000, "voxel_yx_nm": vy * 1000,
        "n_z_slices": n_z, "z_extent_um": round(n_z * vz, 2),
        "n_axons": len(cells), "axon_volume_um3": total_axon_vol,
        "z25_fractions_comparable": False,
    }

    for variant in VARIANTS:
        d = f"{base}/{sid}_morphometrics_imod_{variant}"
        for setname, fname in SETS.items():
            mitos = pd.read_csv(f"{d}/{fname}")
            for k, v in summarise(mitos, total_axon_vol).items():
                row[f"{k}_{variant}_{setname}"] = v

            # per-axon rows, and the mean-of-axons fraction for the full runs
            fracs = []
            for lab, av in sorted(axon_vol.items()):
                sub = mitos[mitos.cell_id == lab]
                rec = summarise(sub, av)
                fracs.append(rec["mito_volume_fraction_pooled"])
                per_axon.append({
                    "animal": sid, "genotype": grp, "variant": variant, "object_set": setname,
                    "axon_label": int(lab), "axon_name": axon_names.get(lab, ""),
                    "axon_volume_um3": av,
                    "mito_count": rec["n_mito"],
                    "mito_volume_um3": rec["mito_volume_um3_total"],
                    "mito_volume_fraction": rec["mito_volume_fraction_pooled"],
                    "mito_count_per_100um3": rec["mito_count_per_100um3"],
                    "median_mito_volume_um3": rec["median_mito_volume_um3"],
                    "median_sphericity": rec["median_sphericity"],
                    "median_elongation": rec["median_elongation"],
                })
            if variant == "full":
                row[f"mito_volume_fraction_mean_of_axons_full_{setname}"] = float(np.mean(fracs))

    # the z25 window, read back from the run parameters so the CSV cannot drift from the run
    with open(f"{base}/{sid}_morphometrics_imod_z25/run_parameters.txt") as f:
        z = [l.split(":", 1)[1].strip() for l in f if l.startswith("z_range:")][0]
    row["z25_window"] = z.replace("-zs ", "").replace(" -ze ", "-")
    overview.append(row)

ov = pd.DataFrame(overview).sort_values(["genotype", "animal"]).reset_index(drop=True)
pa = pd.DataFrame(per_axon).sort_values(
    ["genotype", "animal", "variant", "object_set", "axon_label"]).reset_index(drop=True)

# column order: metadata, then metrics grouped by variant/set
meta = ["animal", "genotype", "source_model", "has_raw_em", "voxel_z_nm", "voxel_yx_nm",
        "n_z_slices", "z_extent_um", "n_axons", "axon_volume_um3", "z25_window",
        "z25_fractions_comparable"]
ordered = meta + [f"{m}_{v}_{s}" for v in VARIANTS for s in SETS for m in METRICS] \
    + [f"mito_volume_fraction_mean_of_axons_full_{s}" for s in SETS]
ov = ov[ordered]

ov.to_csv(f"{DATA_ROOT}/mito_morphometrics_overview.csv", index=False)
pa.to_csv(f"{DATA_ROOT}/mito_morphometrics_per_axon.csv", index=False)
print(f"wrote mito_morphometrics_overview.csv ({len(ov)} rows, {len(ov.columns)} cols)")
print(f"wrote mito_morphometrics_per_axon.csv ({len(pa)} rows)")

# terminal summary: the headline comparison
print("\n=== mitochondrial load, full extent, all traced objects ===")
show = ov[["animal", "genotype", "n_mito_full_all", "axon_volume_um3",
           "mito_volume_fraction_pooled_full_all", "mito_count_per_100um3_full_all",
           "median_mito_volume_um3_full_all", "median_sphericity_full_all"]]
print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

for label, col in (("volume fraction", "mito_volume_fraction_pooled_full_all"),
                   ("count per 100um3", "mito_count_per_100um3_full_all"),
                   ("median mito volume", "median_mito_volume_um3_full_all"),
                   ("median sphericity", "median_sphericity_full_all")):
    ko = ov[ov.genotype == "KO"][col].values
    wt = ov[ov.genotype == "WT"][col].values
    sep = "KO > WT, no overlap" if ko.min() > wt.max() else (
        "KO < WT, no overlap" if ko.max() < wt.min() else "overlapping")
    print(f"{label:20s} KO={np.round(ko, 4)} WT={np.round(wt, 4)}  ratio={ko.mean()/wt.mean():.2f}x  {sep}")


# ---------------------------------------------------------------------------
# Manual (IMOD) vs automatic segmentation, for the two animals that have both.
#
# The automatic run is the published one: 10 hand-picked axons, 25 um z-window.
# Its axon set is NOT the manual 5 -- only 3 of 5 overlap in 4007 and 2 of 5 in
# 4010 -- so the aggregate below compares each analysis as published and its
# densities are not interchangeable. See mito_morphometrics_findings.md.
# ---------------------------------------------------------------------------
cmp_rows, cmp_axon, matched = [], [], []

for sid, (grp, zwin) in AUTO.items():
    base = f"{DATA_ROOT}/{grp}/{sid}"
    auto_dir = f"{base}/{sid}_morphometrics_final"
    if not os.path.exists(f"{auto_dir}/mito_morphometrics_3d.csv"):
        print(f"\n[skip] {sid}: no automatic run at {auto_dir}")
        continue

    man = pd.read_csv(f"{base}/{sid}_morphometrics_imod_full/mito_morphometrics_3d.csv")
    man_cells = pd.read_csv(f"{base}/{sid}_morphometrics_imod_full/cell_summary_3d.csv")
    aut = pd.read_csv(f"{auto_dir}/mito_morphometrics_3d.csv")
    aut_cells = pd.read_csv(f"{auto_dir}/cell_summary_3d.csv")

    # the automatic whitelist is whatever its mito table actually references
    whitelist = sorted(aut.cell_id.dropna().astype(int).unique())
    aut_cells = aut_cells[aut_cells.label_id.isin(whitelist)]

    g = zarr.open(f"{base}/{sid}_imod.zarr", "r")
    n_z = g["mitos"].shape[0]
    vz = g.attrs["voxel_size"][0]

    for pipeline, mitos, cells, aset, zf, zl in (
        ("imod", man, man_cells, f"{len(man_cells)} manually traced", 0, n_z - 1),
        ("auto", aut, aut_cells, f"{len(aut_cells)} whitelisted, automatic seg", zwin[0], zwin[1]),
    ):
        vol = float(cells.volume_um3.sum())
        rec = summarise(mitos, vol)
        cmp_rows.append({
            "animal": sid, "genotype": grp, "pipeline": pipeline, "axon_set": aset,
            "n_axons": len(cells), "axon_volume_um3": vol,
            "n_mito": rec["n_mito"], "mito_volume_um3_total": rec["mito_volume_um3_total"],
            "mito_volume_fraction": rec["mito_volume_fraction_pooled"],
            "mito_count_per_100um3": rec["mito_count_per_100um3"],
            "median_mito_volume_um3": rec["median_mito_volume_um3"],
            "median_sphericity": rec["median_sphericity"],
            "median_elongation": rec["median_elongation"],
            "z_first": zf, "z_last": zl, "z_slices": zl - zf + 1,
            "z_extent_um": round((zl - zf + 1) * vz, 2),
            # different axon sets + different z coverage + the automatic numerator is
            # z-windowed while its denominator is full-extent
            "density_comparable": False,
        })

    # which manual axon does each automatic mitochondrion sit in? centroid lookup in
    # the manual axon labels -- needs no automatic axon segmentation, so 4007 works too
    ax = g["axons"]
    zz = np.clip(aut.centroid_z_vx.round().astype(int), 0, ax.shape[0] - 1)
    yy = np.clip(aut.centroid_y_vx.round().astype(int), 0, ax.shape[1] - 1)
    xx = np.clip(aut.centroid_x_vx.round().astype(int), 0, ax.shape[2] - 1)
    aut = aut.assign(manual_axon_at_centroid=[int(ax[a, b, c]) for a, b, c in zip(zz, yy, xx)])

    # an automatic axon counts as covering a manual axon if most of its mitos land there
    auto_to_manual = {}
    for aid, sub in aut.groupby(aut.cell_id.astype(int)):
        hits = sub.manual_axon_at_centroid[sub.manual_axon_at_centroid > 0]
        auto_to_manual[aid] = int(hits.mode().iloc[0]) if len(hits) > len(sub) / 2 else 0
    covered = {v for v in auto_to_manual.values() if v > 0}

    # per axon, per pipeline
    for pipeline, mitos, cells in (("imod", man, man_cells), ("auto", aut, aut_cells)):
        for _, crow in cells.iterrows():
            lab = int(crow.label_id)
            sub = mitos[mitos.cell_id == lab]
            rec = summarise(sub, crow.volume_um3)
            matched_manual = (lab if pipeline == "imod" else auto_to_manual.get(lab, 0))
            cmp_axon.append({
                "animal": sid, "genotype": grp, "pipeline": pipeline, "axon_id": lab,
                "overlapping_manual_axon": matched_manual or "",
                "axon_volume_um3": crow.volume_um3,
                "mito_count": rec["n_mito"], "mito_volume_um3": rec["mito_volume_um3_total"],
                "mito_volume_fraction": rec["mito_volume_fraction_pooled"],
                "mito_count_per_100um3": rec["mito_count_per_100um3"],
                "median_mito_volume_um3": rec["median_mito_volume_um3"],
            })

    # per-mito matching: nearest automatic centroid to each manual one
    C = ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
    dist, idx = cKDTree(aut[C].values).query(man[C].values)
    for (_, m), d, i in zip(man.iterrows(), dist, idx):
        a = aut.iloc[i]
        is_match = bool(d < MATCH_TOL_UM)
        matched.append({
            "animal": sid, "genotype": grp,
            "manual_label_id": int(m.label_id), "manual_axon": int(m.cell_id),
            "manual_axon_in_auto_whitelist": int(m.cell_id) in covered,
            "manual_volume_um3": m.volume_um3,
            "auto_label_id": int(a.label_id), "auto_volume_um3": a.volume_um3,
            "centroid_distance_um": float(d),
            # only meaningful for a real match: for a non-match the nearest automatic
            # object is a different mitochondrion, so the ratio would be nonsense
            "volume_ratio_auto_over_manual": float(a.volume_um3 / m.volume_um3) if is_match else np.nan,
            "matched_within_0p5um": is_match,
        })

if cmp_rows:
    cdf = pd.DataFrame(cmp_rows).sort_values(["animal", "pipeline"]).reset_index(drop=True)
    adf = pd.DataFrame(cmp_axon).sort_values(["animal", "pipeline", "axon_id"]).reset_index(drop=True)
    mdf = pd.DataFrame(matched).sort_values(["animal", "manual_label_id"]).reset_index(drop=True)
    cdf.to_csv(f"{DATA_ROOT}/mito_morphometrics_auto_comparison.csv", index=False)
    adf.to_csv(f"{DATA_ROOT}/mito_morphometrics_auto_per_axon.csv", index=False)
    mdf.to_csv(f"{DATA_ROOT}/mito_morphometrics_auto_matched_mitos.csv", index=False)
    print(f"\nwrote mito_morphometrics_auto_comparison.csv ({len(cdf)} rows)")
    print(f"wrote mito_morphometrics_auto_per_axon.csv ({len(adf)} rows)")
    print(f"wrote mito_morphometrics_auto_matched_mitos.csv ({len(mdf)} rows)")

    print("\n=== manual (IMOD) vs automatic, each as published ===")
    print("NOTE different axon sets and different z coverage -- densities are not interchangeable")
    print(cdf[["animal", "genotype", "pipeline", "n_axons", "axon_volume_um3", "n_mito",
               "mito_count_per_100um3", "mito_volume_fraction", "median_mito_volume_um3",
               "z_extent_um"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== do the two pipelines find the same mitochondria? ===")
    for sid in sorted(mdf.animal.unique()):
        s = mdf[mdf.animal == sid]
        cov = s[s.manual_axon_in_auto_whitelist]
        print(f"{sid}: {s.matched_within_0p5um.mean():.0%} of all {len(s)} manual mitos matched "
              f"(NOT recall -- most misses are axons the automatic whitelist never covered)")
        print(f"      {cov.matched_within_0p5um.mean():.0%} of the {len(cov)} in shared axons "
              f"{sorted(cov.manual_axon.unique())}  <- the meaningful figure")
        mm = cov[cov.matched_within_0p5um]
        if len(mm):
            print(f"      volume agreement on those: median auto/manual ratio "
                  f"{mm.volume_ratio_auto_over_manual.median():.2f}, "
                  f"median centroid offset {mm.centroid_distance_um.median():.3f} um")

    print("\n=== KO vs WT, does the automatic data agree in direction? ===")
    for pipeline in ("imod", "auto"):
        p = cdf[cdf.pipeline == pipeline].set_index("animal")
        ko, wt = p.loc["4007"], p.loc["4010"]
        for met in ("mito_volume_fraction", "mito_count_per_100um3"):
            arrow = "KO > WT" if ko[met] > wt[met] else "KO < WT"
            print(f"  {pipeline:4s} {met:22s} KO 4007={ko[met]:8.4f}  WT 4010={wt[met]:8.4f}  "
                  f"{arrow}  ratio={ko[met]/wt[met]:.2f}x")
