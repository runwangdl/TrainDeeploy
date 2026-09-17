#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Size the readfs tail-padding file so the GAP9 EVK flash image ends in a safe spot.

On the GAP9 EVK the last readfs file(s) of `flash.bin` read back corrupted when
the image ends 48-64 KB into a 64 KB sector (observed 2026-09-14: image sizes
mod 64 KB of 61440 -> CCT promoted weights / MobileNetV1 mini-batch 3 corrupted;
<= 36864 always intact). A dummy file that sorts last in the readfs
(`zzz_readfs_pad.hex`) absorbs the tail: this script resizes it so the final
image lands ~20 KB into a sector, and `--check` refuses an image in the bad zone.

usage: readfs_pad.py resize <flash.bin> <padfile>   # rewrite padfile, exit 0
       readfs_pad.py check  <flash.bin>             # exit 1 if in the bad zone
"""
import os
import sys

SECTOR = 64 * 1024
TARGET_IN_SECTOR = 20 * 1024  # where the image should end, measured into a sector
BAD_FROM = 48 * 1024  # empirically: >= 48 KB into the sector => tail corrupted
MIN_PAD = SECTOR  # keep every real file at least one sector away from the end


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    image = sys.argv[2]
    size = os.path.getsize(image)
    into = size % SECTOR
    if mode == "check":
        print(f"[readfs_pad] {os.path.basename(image)} = {size} B, {into} B into its last 64 KB sector")
        if into >= BAD_FROM:
            print("[readfs_pad] ERROR: image ends in the zone where the EVK returns a corrupted "
                  "readfs tail (>= 48 KB into a sector). Enable/resize the readfs pad file.")
            return 1
        return 0
    if mode == "resize":
        padfile = sys.argv[3]
        cur = os.path.getsize(padfile) if os.path.exists(padfile) else 0
        # size of the image without the current pad payload, then choose the pad
        base = size - cur
        need = MIN_PAD + (TARGET_IN_SECTOR - (base + MIN_PAD) % SECTOR) % SECTOR
        need = max(MIN_PAD, need // 4096 * 4096)
        with open(padfile, "wb") as f:
            f.write(b"Z" * need)
        print(f"[readfs_pad] image {size} B (pad was {cur} B) -> pad {need} B, "
              f"expected image end {(base + need) % SECTOR} B into its sector")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
