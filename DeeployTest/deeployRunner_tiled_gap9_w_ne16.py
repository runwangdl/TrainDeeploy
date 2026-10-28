#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# -- QW: tiled inference runner for GAP9 + the NE16 accelerator engine.
#        Same as deeployRunner_tiled_gap9.py with default_platform="GAP9_w_NE16".
#        `--enable-3x3` extends NE16Engine.canExecute to 3x3 dense/depthwise convs; 1x1
#        pointwise is always accepted, so STEP 2a does not need the flag.

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    def setup_parser(parser):  # -- QW
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')
        parser.add_argument('--enable-1xk',
                            dest = 'enable_1xk',
                            action = 'store_true',
                            default = False,
                            help = 'Let NE16 claim 1xK / Kx1 dense convs and run them as K '
                            'pointwise dispatches accumulating via streamin (exp16a / STEP 2b)\n')
        parser.add_argument('--enable-3x3',
                            dest = 'enable_3x3',
                            action = 'store_true',
                            default = False,
                            help = 'Extend NE16 to 3x3 dense and 3x3 depthwise convolutions\n')

    sys.exit(
        main(default_platform = "GAP9_w_NE16",
             default_simulator = "gvsoc",
             tiling_enabled = True,
             parser_setup_callback = setup_parser))
