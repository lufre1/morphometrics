#!/usr/bin/env bash
# 3D morphometrics for the manually annotated (IMOD-derived) label volumes.
#
# These volumes have no raw EM data, so morphometrics_3d.py runs without --path:
# the four intensity columns are omitted and the mean_intensity QC gate is skipped.
# All geometric metrics come from the label volumes alone and are unaffected.
#
# Two z-ranges per animal: the full traced extent, and a 25 um window centred in z.
# -zs/-ze restricts the WHOLE analysis to the window: mitos crossing either face are
# dropped (not truncated, so shape metrics stay valid), and the axon volume counts only
# in-window voxels. Numerator and denominator therefore describe the same tissue -- before
# 2026-09 the flags filtered mitos only, leaving every z25 density biased low.
#
# Set VARIANTS to rerun a subset, e.g. VARIANTS=z25 ./run_morphometrics_imod.sh
#
# Run inside the repo env: `micromamba activate mito-morphometrics` (see env.yaml).
set -euo pipefail

R=/home/freckmann15/data/volume-em
REPO="$(cd "$(dirname "$0")" && pwd)"
S="$REPO/morphometrics_3d.py"
BLOCK="64 1024 1024"   # bigger than default: fewer, larger block reads over sparse labels

# id group voxel_size_zyx_nm z25_start z25_end
SAMPLES=(
  "4005 KO 25 5 5 168 1167"
  "4007 KO 25 5 5 224 1223"
  "4009 WT 25 5 5  27 1026"
  "4010 WT 25 5 5 276 1275"
  "4015 KO 10 5 5  34 2533"   # IMOD z-scale 2 -> 10 nm z step, so 25 um = 2500 slices
  "4016 WT 25 5 5 125 1124"
)

for s in "${SAMPLES[@]}"; do
  read -r ID GRP VZ VY VX ZS ZE <<< "$s"
  Z="$R/$GRP/$ID/${ID}_imod.zarr"

  for VARIANT in ${VARIANTS:-full z25}; do
    OUT="$R/$GRP/$ID/${ID}_morphometrics_imod_${VARIANT}"
    if [ "$VARIANT" = "z25" ]; then ZARGS=(-zs "$ZS" -ze "$ZE"); else ZARGS=(); fi

    echo "=== $ID ($GRP) $VARIANT ==="
    python "$S" \
      --mito_label_path "$Z" --mito_key mitos \
      --cell_label_path "$Z" --cell_key axons \
      --voxel_size "$VZ" "$VY" "$VX" \
      --no_cell_surface \
      --block_shape $BLOCK \
      --output_path "$OUT" "${ZARGS[@]}"

    {
      echo "sample:        $ID ($GRP)"
      echo "source:        $Z  (mito_key=mitos, cell_key=axons)"
      echo "voxel_size_nm: $VZ $VY $VX  (z y x)"
      echo "z_range:       ${ZARGS[*]:-full traced extent}"
      echo "raw:           none (label-only run; intensity columns omitted)"
      echo "flags:         --no_cell_surface --block_shape $BLOCK"
      echo "generated:     $(date -Is)"
    } > "$OUT/run_parameters.txt"

    # mirror into the repo data/ dir, where figures_3d.py reads from
    rm -rf "$REPO/data/${ID}_morphometrics_imod_${VARIANT}"
    cp -r "$OUT" "$REPO/data/${ID}_morphometrics_imod_${VARIANT}"
  done
done

echo "All runs complete."
