# ----------------------------------------------------------------------
#
# File: Engine.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import ConvGradWLayer, ConvGradXLayer, ConvLayer, GEMMLayer, MatMulLayer
from Deeploy.Targets.Generic.Parsers import MatMulParser
from Deeploy.Targets.PULPOpen.Parsers import PULPFPConv2DParser, PULPPWConvGradW2DParser, PULPPWConvGradX2DParser
from Deeploy.Targets.Redmule.Parsers import GEMMRedmuleParser
from Deeploy.Targets.Redmule.Tiler import RedmuleConvTilingReadyBindings, RedmuleGEMMTilingReadyBindings, \
    RedmuleMatMulTilingReadyBindings, RedmulePWConvGradW2DTilingReadyBindings, RedmulePWConvGradX2DTilingReadyBindings

MatMulRedmuleMapper = NodeMapper(MatMulParser(), RedmuleMatMulTilingReadyBindings)
Conv2DRedmuleMapper = NodeMapper(PULPFPConv2DParser(), RedmuleConvTilingReadyBindings)
GEMMMRedmuleMapper = NodeMapper(GEMMRedmuleParser(noBiasHoisting = False), RedmuleGEMMTilingReadyBindings)
# Pointwise (1x1) ConvGradW / ConvGradX reuse PULP's parsers verbatim --
# they only screen for kernel_shape=[1,1] / group=1 and populate the same
# operatorRepresentation keys our Redmule templates consume.
PWConvGradW2DRedmuleMapper = NodeMapper(PULPPWConvGradW2DParser(), RedmulePWConvGradW2DTilingReadyBindings)
PWConvGradX2DRedmuleMapper = NodeMapper(PULPPWConvGradX2DParser(), RedmulePWConvGradX2DTilingReadyBindings)

RedmuleMapping = {
    'MatMul': MatMulLayer([MatMulRedmuleMapper]),
    # 'Conv' is currently routed to PULPClusterEngine (see comment below).
    # The RedMulE-accelerated kernel and its template are kept in-tree
    # (TargetLibraries/PULPOpen/src/Conv2d_Im2Col_fp32_Redmule.c +
    # Deeploy/Targets/Redmule/Templates/ConvTemplate.py) so the path is
    # ready to re-enable once RedmuleConv2DTileConstraint learns spatial
    # tiling with halo regions.  Today its addPolicyConstraint hard-pins
    # inputHeightVar / inputWidthVar to the full feature-map size, which
    # forces the entire activation tensor into L1 -- workable for tiny
    # tokenizer-style Convs (CCT2 has 8x8 inputs and L1=128 KiB fits),
    # but ResNet8 / MobileNet middle layers exceed L1 immediately
    # (32x32x16 input + 32x32x16 output alone is 128 KiB).  PULP's
    # Conv2DTileConstraint already supports spatial halos, so falling
    # back keeps the bigger Conv-heavy training fixtures tilable while
    # MatMul / Gemm continue to bind to RedMulE.
    #
    # When that tile-constraint upgrade lands, restore:
    #     'Conv': ConvLayer([Conv2DRedmuleMapper]),
    # and the matching RedMuleAdjustWeightMemoryLayoutPass in Deployer.py.
    'Gemm': GEMMLayer([GEMMMRedmuleMapper]),
    # Pointwise (1x1) ConvGrad: only the PW variant is mapped here.  The
    # PULPPWConvGrad*Parsers require kernel_shape == [1, 1] and group == 1,
    # so non-PW backward Convs (regular 3x3 ConvGradW, depthwise variants)
    # transparently fall through to PULPClusterEngine -- which carries the
    # full [PW, DW, regular] mapper list.
    'ConvGradW': ConvGradWLayer([PWConvGradW2DRedmuleMapper]),
    'ConvGradX': ConvGradXLayer([PWConvGradX2DRedmuleMapper]),
}

_includeList = []

_redmuleInitCode = r"""
// Redmule engine initialization
"""


class RedmuleEngine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = RedmuleMapping,
                 initCode: str = _redmuleInitCode,
                 includeList: List[str] = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)
