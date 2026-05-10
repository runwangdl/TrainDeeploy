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
from Deeploy.Targets.PULPOpen.Platform import ConvGradWMapper as PULPConvGradWMapper, \
    ConvGradXMapper as PULPConvGradXMapper, DwConvGradWMapper, DwConvGradxMapper
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
    # NOTE: ConvGradW / ConvGradX are intentionally NOT mapped here.
    # _selectEngine() is first-match across engines, so putting them on the
    # RedmuleEngine would route every 3x3 / depthwise ConvGrad through this
    # engine's layer and never let PULPClusterEngine see them.  We tried a
    # "complete" RedmuleEngine layer ([PW_Redmule, DW_PULP, regular_PULP])
    # but the resulting tiler hit infeasible memory-pattern constraints on
    # ResNet8 / MobileNet despite using identical mapper instances to
    # PULP -- some interaction between the layer object identity and the
    # tiling-pattern solver we couldn't fully diagnose.
    #
    # Instead, the RedMulE PWConvGrad mappers are inserted into the
    # existing PULPClusterEngine ConvGradW / ConvGradX layers at position 0
    # in RedmulePlatform.__init__.  That keeps the layer object identical
    # to the pure-PULP path (matters for the tiler) while still ensuring
    # 1x1 ConvGrads bind to the RedMulE kernels.
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
