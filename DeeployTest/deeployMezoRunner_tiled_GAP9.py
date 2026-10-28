#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# -- QW: thin entry point for the tiled GAP9 MeZO (ZO / QZO) training runner.
#        Identical to deeployMezoRunner_tiled_siracusa.py except the target
#        platform is wired to GAP9 (same QZO fixture, harness and hyperparameters).

import sys

from testUtils.deeployMezoRunner import main  # -- QW

if __name__ == '__main__':
    sys.exit(main(tiling_enabled = True, default_platform = 'GAP9'))  # -- QW
