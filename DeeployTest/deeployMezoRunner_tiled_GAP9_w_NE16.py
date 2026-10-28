#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# -- QW: thin entry point for the tiled GAP9+NE16 MeZO (ZO / QZO) training runner.
#        Identical to deeployMezoRunner_tiled_GAP9.py except the target platform is
#        GAP9_w_NE16 (GAP9 cluster + the NE16 accelerator engine).
#
#        NOTE: deeployMezoRunner.main() does `platform = default_platform` and IGNORES
#        args.platform, so a separate runner file is the intended way to select a platform
#        (same pattern as deeployRunner_tiled_gap9_w_ne16.py upstream). Any NE16-specific
#        flag must be threaded explicitly into gen_args inside deeployMezoRunner.main(),
#        or it silently does nothing.

import sys

from testUtils.deeployMezoRunner import main  # -- QW

if __name__ == '__main__':
    sys.exit(main(tiling_enabled = True, default_platform = 'GAP9_w_NE16'))  # -- QW
