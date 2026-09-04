"""Verification of the IMOD-derived morphometrics runs (plan Step 3, checks 1-6)."""
import glob
import os

import numpy as np
import pandas as pd
import zarr

R = "/home/freckmann15/data/volume-em"
GROUP = {"4005": "KO", "4007": "KO", "4015": "KO", "4009": "WT", "4010": "WT", "4016": "WT"}
fail = []
ELONG = {}


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fail.append(msg)


for sid, grp in sorted(GROUP.items()):
    base = f"{R}/{grp}/{sid}"
    objs = pd.read_csv(f"{base}/{sid}_imod_objects.csv")
    mit_obj = objs[objs.key == "mitos"]
    ax_obj = objs[objs.key == "axons"]
    vox = zarr.open(f"{base}/{sid}_imod.zarr", "r").attrs["voxel_size"]  # um, zyx
    full_dir, z25_dir = f"{base}/{sid}_morphometrics_imod_full", f"{base}/{sid}_morphometrics_imod_z25"
    if not os.path.exists(f"{full_dir}/mito_morphometrics_3d.csv"):
        print(f"\n{sid} ({grp}): not run yet, skipping")
        continue

    print(f"\n{sid} ({grp})  voxel_size_um={vox}")
    m = pd.read_csv(f"{full_dir}/mito_morphometrics_3d.csv")
    c = pd.read_csv(f"{full_dir}/cell_summary_3d.csv")
    qc = pd.read_csv(f"{full_dir}/mito_morphometrics_3d_qc.csv")

    # 2. nothing lost
    check(len(m) == len(mit_obj), f"mito rows {len(m)} == exported mitos {len(mit_obj)}")
    check(len(c) == len(ax_obj), f"cell rows {len(c)} == exported axons {len(ax_obj)}")

    # 1. cell assignment matches the IMOD object names
    named = mit_obj.dropna(subset=["parent_axon"])
    j = m.merge(named[["label", "parent_axon", "name"]], left_on="label_id", right_on="label")
    mism = j[j.cell_id != j.parent_axon]
    # 4015 label 70 is named "2 Mito 12" (a duplicate of label 50) but lies 100% inside
    # axon 3, verified by exhaustive voxel census: the source annotation name is wrong,
    # the overlap-based assignment is right.
    known_bad = {("4015", 70)}
    mism = mism[~mism.label_id.map(lambda l: (sid, int(l)) in known_bad)]
    check(len(mism) == 0,
          f"cell_id matches parent_axon for all {len(j)} named mitos"
          + ("" if len(mism) == 0 else f" -- mismatches: {mism[['name','cell_id','parent_axon']].values.tolist()[:5]}"))
    unassigned = m.cell_id.isna().sum()
    check(unassigned == 0, f"every mito assigned to an axon (unassigned={unassigned})")

    # 3. voxel size handling: volume_um3 == n_voxels * voxel volume
    vv = float(np.prod(vox))
    jm = m.merge(mit_obj[["label", "n_voxels"]], left_on="label_id", right_on="label")
    check(np.allclose(jm.volume_um3, jm.n_voxels * vv, rtol=1e-9),
          f"mito volume_um3 == n_voxels * {vv:.3e} um^3")
    jc = c.merge(ax_obj[["label", "n_voxels"]], left_on="label_id", right_on="label")
    check(np.allclose(jc.volume_um3, jc.n_voxels * vv, rtol=1e-9), "cell volume_um3 == n_voxels * voxel volume")

    # 5. QC: intensity columns absent, gate skipped; border filter only
    check("mean_intensity" not in m.columns, "no intensity columns (label-only run)")
    # the two active gates: border, and elongation >= 20 (the intensity gate is skipped)
    border = int(m.touches_border.sum())
    elong = int(((m.elongation_a_over_c >= 20) & ~m.touches_border).sum())
    check(len(qc) == len(m) - border - elong,
          f"QC = {border} border + {elong} elongation>=20 ({len(m)} -> {len(qc)})")
    ELONG[sid] = elong

    # 4. z25 subset of full
    if os.path.exists(f"{z25_dir}/mito_morphometrics_3d.csv"):
        m25 = pd.read_csv(f"{z25_dir}/mito_morphometrics_3d.csv")
        check(set(m25.label_id) <= set(m.label_id), f"z25 label set subset of full ({len(m25)} of {len(m)} mitos)")
    else:
        print("  ..   z25 not run yet")

    print(f"       mito volume_um3: median={m.volume_um3.median():.4f} "
          f"range=[{m.volume_um3.min():.4f}, {m.volume_um3.max():.4f}]  "
          f"sphericity median={m.sphericity.median():.3f}")

# 6. manual vs automatic, where the automatic result CSVs exist
print("\n=== manual vs automatic (same tissue, independent pipelines) ===")
for sid, grp in (("4010", "WT"), ("4007", "KO")):
    man_p = f"{R}/{grp}/{sid}/{sid}_morphometrics_imod_full/mito_morphometrics_3d.csv"
    aut_p = f"{R}/{grp}/{sid}/{sid}_morphometrics_final/mito_morphometrics_3d.csv"
    if not (os.path.exists(man_p) and os.path.exists(aut_p)):
        print(f"{sid}: missing one side, skipping")
        continue
    man, aut = pd.read_csv(man_p), pd.read_csv(aut_p)
    print(f"{sid}: manual n={len(man):4d} median={man.volume_um3.median():.4f} um3 | "
          f"automatic n={len(aut):4d} median={aut.volume_um3.median():.4f} um3 | "
          f"ratio={man.volume_um3.median() / aut.volume_um3.median():.2f}")

print(f"\nmitos removed by the elongation>=20 gate (tuned for automatic merge artefacts): {ELONG}")
print("\n" + ("ALL CHECKS PASSED" if not fail else f"{len(fail)} FAILURES:\n  " + "\n  ".join(fail)))
