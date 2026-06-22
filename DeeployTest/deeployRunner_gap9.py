#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add GAP9-specific arguments
    def setup_parser(parser):
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')
        parser.add_argument('--convChannelsFirst',
                            action = 'store_true',
                            default = False,
                            help = 'Keep forward convs channels-first (NCHW), binding the *_CHW kernels.\n')
        parser.add_argument('--powerMeasurement',
                            action = 'store_true',
                            default = False,
                            help = 'Enable GPIO toggling around the inference window for external '
                            'power measurement (e.g. PPK2). Only meaningful with -s board.\n')
        parser.add_argument('--freqFC',
                            type = int,
                            default = 370,
                            metavar = 'MHz',
                            help = 'FC clock frequency in MHz for power measurement (default: 370).\n')
        parser.add_argument('--freqCL',
                            type = int,
                            default = 370,
                            metavar = 'MHz',
                            help = 'Cluster clock frequency in MHz for power measurement (default: 370).\n')
        parser.add_argument('--freqPE',
                            type = int,
                            default = 370,
                            metavar = 'MHz',
                            help = 'Peripheral clock frequency in MHz for power measurement (default: 370).\n')

    sys.exit(
        main(default_platform = "GAP9",
             default_simulator = "gvsoc",
             tiling_enabled = False,
             parser_setup_callback = setup_parser))
