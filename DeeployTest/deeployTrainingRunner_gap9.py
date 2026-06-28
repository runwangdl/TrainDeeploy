#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import sys

from testUtils.deeployTrainingRunner import main

if __name__ == '__main__':

    # GAP9-specific arguments (--cores is already added by the common runner).
    def setup_parser(parser):
        parser.add_argument('--convChannelsFirst',
                            action = 'store_true',
                            default = False,
                            help = 'Keep forward convs channels-first (NCHW), binding the *_CHW kernels '
                            '(removes the NCHW<->NHWC transpose around every conv; required for '
                            'MobileNetV1 training to fit GAP9 L1).\n')
        parser.add_argument('--powerMeasurement',
                            action = 'store_true',
                            default = False,
                            help = 'Enable GPIO 89 toggling around the training loop for PPK2 power '
                            'measurement. Only meaningful with -s board.\n')
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
        main(tiling_enabled = False,
             default_platform = "GAP9",
             default_simulator = "gvsoc",
             parser_setup_callback = setup_parser))
