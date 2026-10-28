#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# -- QW: thin entry point for the tiled Siracusa MeZO (ZO) training runner.

import sys

from testUtils.deeployMezoRunner import main  # -- QW

if __name__ == '__main__':
    sys.exit(main(tiling_enabled = True))  # -- QW
