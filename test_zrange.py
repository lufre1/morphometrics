#!/usr/bin/env python
"""Self-check for the z-window restriction in morphometrics_3d.

The block loop in `_scan_bboxes` clips each block to the window and offsets coordinates by
the clipped start. That is exactly the kind of arithmetic where an off-by-one survives
silently and only shows up as a slightly wrong density after hours of recompute, so it gets
a check that runs in a second.

    micromamba run -n synapse python test_zrange.py
"""

import numpy as np
import zarr

from morphometrics_3d import _scan_bboxes


def _volume(tmp):
    """A 20-slice volume with three labels placed relative to the window z in [5, 14].

    label 1  z 6..9    fully inside
    label 2  z 3..7    crosses the lower face
    label 3  z 12..17  crosses the upper face
    label 4  z 0..2    fully outside
    """
    vol = np.zeros((20, 4, 4), dtype=np.uint16)
    vol[6:10, 0:2, 0:2] = 1
    vol[3:8, 2:4, 0:2] = 2
    vol[12:18, 0:2, 2:4] = 3
    vol[0:3, 2:4, 2:4] = 4
    g = zarr.open(tmp, mode="w")
    g.create_dataset("seg", data=vol, chunks=(8, 4, 4))
    return vol


def main():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/v.zarr"
        vol = _volume(path)
        window = (5, 14)
        # block_shape z=8 does not divide the window, so blocks are clipped on both faces
        # and the partial-block path is actually exercised.
        bs = (8, 4, 4)

        full_bb, full_vc, shape = _scan_bboxes(path, "seg", bs)
        win_bb, win_vc, _ = _scan_bboxes(path, "seg", bs, z_range=window)

        assert shape == (20, 4, 4), shape

        # 1. an unwindowed scan is unchanged: it must equal a plain voxel count
        for lid in (1, 2, 3, 4):
            assert full_vc[lid] == int((vol == lid).sum()), (lid, full_vc[lid])
        assert full_bb[1] == [6, 9, 0, 1, 0, 1], full_bb[1]

        # 2. windowed voxel counts equal the hand-counted in-window voxels
        zlo, zhi = window
        inside = np.zeros_like(vol)
        inside[zlo:zhi + 1] = vol[zlo:zhi + 1]
        for lid in (1, 2, 3):
            assert win_vc[lid] == int((inside == lid).sum()), (lid, win_vc[lid])

        # 3. a label with no in-window voxel is absent entirely, not present with count 0.
        #    This is what drops out-of-window axons from the per-cell summary.
        assert 4 not in win_vc and 4 not in win_bb, sorted(win_vc)

        # 4. windowed bboxes are clipped to the window -- the reason the mito scan must
        #    NOT be windowed, since a clipped straddler would compare as contained.
        assert win_bb[2][0] == zlo, win_bb[2]      # label 2 clipped at the lower face
        assert win_bb[3][1] == zhi, win_bb[3]      # label 3 clipped at the upper face
        assert win_bb[1] == full_bb[1], win_bb[1]  # the interior one is untouched

        # 5. the containment filter, applied to TRUE bboxes, keeps only the interior label
        kept = [lid for lid in sorted(full_bb)
                if full_bb[lid][0] >= zlo and full_bb[lid][1] <= zhi]
        assert kept == [1], kept

        # 6. and the same filter on the CLIPPED bboxes wrongly keeps the straddlers --
        #    the bug this asymmetry exists to avoid.
        wrong = [lid for lid in sorted(win_bb)
                 if win_bb[lid][0] >= zlo and win_bb[lid][1] <= zhi]
        assert wrong == [1, 2, 3], wrong

    print("test_zrange: all checks passed")


if __name__ == "__main__":
    main()
