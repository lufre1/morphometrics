#!/usr/bin/env bash
# 3D morphometrics for the automatically segmented (CNN) volumes -- the "auto" arm.
#
# Why this exists: the published <id>_morphometrics_final/ tables were produced in
# 2026-05 by a script version whose -zs/-ze restricted the MITO LIST only. Two
# consequences, both measured against the segmentations:
#
#   1. cell_volume_um3 is the full traced extent, not the 25 um window -- identical to
#      the unrestricted <id>_morphometrics_all_final/ run, axon for axon. Every density
#      and volume fraction is therefore low by 1.13-1.58x (mean 1.42x in both animals).
#   2. mitos were kept on bbox OVERLAP, so objects crossing a window face are included
#      whole. In 4010 that is 28 of 242 objects carrying 21.5% of the mito volume. The
#      QC table does not remove them: touches_border means the edge of the VOLUME, not
#      of the window, so 24 of the 28 survive into mito_morphometrics_3d_qc.csv.
#
# The manual arm (run_morphometrics_imod.sh --variant z25) has used the current, fixed
# semantics since 2026-09: containment, and an axon volume cropped to the same window.
# Re-running the auto arm here puts both pipelines on one code path, so a manual-vs-auto
# difference is the segmentation and nothing else.
#
# Output goes to <id>_morphometrics_auto_z25, NOT over <id>_morphometrics_final: the old
# tables are what the existing READMEs and figures describe, and you want to diff them.
# Expect counts down ~10%, mito volume down ~18%, axon volume down ~1.4x, so densities up
# ~25-35%.
#
# Set ONLY to run one animal, e.g. ONLY=4007 ./run_morphometrics_auto.sh
#
# Run inside the repo env: `micromamba activate mito-morphometrics` (see env.yaml).
set -euo pipefail

R=/home/freckmann15/data/volume-em
M=/home/freckmann15/data/mobie
REPO="$(cd "$(dirname "$0")" && pwd)"
S="$REPO/morphometrics_3d.py"
BLOCK="64 1024 1024"   # bigger than default: fewer, larger block reads over sparse labels

# The z-windows and axon whitelists are the published ones, so the new tables stay
# comparable to the <id>_morphometrics_final READMEs.
#
# cell_key is s2 for 4007 because that is the resolution the published run used: axon
# volumes recomputed from s2 match its cell_volume_um3 to all printed digits. axons_new
# also exists at s0; switching this one word would change the volumes for a second
# reason and make the old-vs-new diff uninterpretable.
run_auto() {
  local ID=$1 GRP=$2 RAW=$3 RAWKEY=$4 MITO=$5 MITOKEY=$6 CELL=$7 CELLKEY=$8 ZS=$9 ZE=${10}
  shift 10
  local CIDS=("$@")

  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$ID" ]; then return 0; fi

  # The 4007 mito segmentation was still transferring when this script was written, so a
  # missing input is a skip with a message, not a set -e abort half way through 4010.
  local p
  for p in "$RAW" "$MITO" "$CELL"; do
    if [ ! -e "$p" ]; then
      echo "=== $ID ($GRP) SKIPPED -- missing input: $p"
      return 0
    fi
  done

  local OUT="$R/$GRP/$ID/${ID}_morphometrics_auto_z25"
  echo "=== $ID ($GRP) auto z25 ==="
  python "$S" \
    --path "$RAW" --raw_key "$RAWKEY" \
    --mito_label_path "$MITO" --mito_key "$MITOKEY" \
    --cell_label_path "$CELL" --cell_key "$CELLKEY" \
    --voxel_size 25 5 5 \
    --no_cell_surface \
    --block_shape $BLOCK \
    -zs "$ZS" -ze "$ZE" \
    -cids "${CIDS[@]}" \
    --output_path "$OUT"

  {
    echo "sample:        $ID ($GRP)"
    echo "pipeline:      automatic (CNN segmentation)"
    echo "raw:           $RAW  (key=$RAWKEY)"
    echo "mito:          $MITO  (key=$MITOKEY)"
    echo "cell:          $CELL  (key=$CELLKEY)"
    echo "voxel_size_nm: 25 5 5  (z y x)"
    echo "z_range:       -zs $ZS -ze $ZE"
    echo "cell_ids:      ${CIDS[*]}"
    echo "flags:         --no_cell_surface --block_shape $BLOCK"
    echo "semantics:     mitos fully inside the window; axon volume cropped to the window"
    echo "supersedes:    ${ID}_morphometrics_final (window applied to the mito list only)"
    echo "generated:     $(date -Is)"
  } > "$OUT/run_parameters.txt"

  # mirror into the repo data/ dir, where the figure scripts read from
  rm -rf "$REPO/data/${ID}_morphometrics_auto_z25"
  cp -r "$OUT" "$REPO/data/${ID}_morphometrics_auto_z25"
}

run_auto 4010 WT \
  "$R/WT/4010/4010.zarr" s0 \
  "$R/WT/4010/grid_search_mitos/sd1_bp10_ft07_bt012.zarr" seg \
  "$R/WT/4010/axons_small_model_corrected_new_v4/axons.zarr" seg \
  252 1252 \
  39 106 227 749 1908 1909 1910 1913 1917 1922

# The mito path below is where 4010 keeps its segmentation; point it at wherever the
# 4007 transfer actually lands. Nothing else about this row should need to change.
run_auto 4007 KO \
  "$M/project_4007/4007/images/ome-zarr/raw.ome.zarr" s0 \
  "$R/KO/4007/grid_search_mitos/sd1_bp10_ft07_bt012.zarr" seg \
  "$M/project_4007/4007/images/ome-zarr/axons_new.ome.zarr" s2 \
  347 1347 \
  106 107 130 200 220 395 396 398 407 806

echo "All runs complete."
