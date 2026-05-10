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
from Deeploy.Targets.Generic.Layers import ConvLayer, GEMMLayer, MatMulLayer
from Deeploy.Targets.Generic.Parsers import MatMulParser
from Deeploy.Targets.PULPOpen.Parsers import PULPFPConv2DParser
from Deeploy.Targets.Redmule.Parsers import GEMMRedmuleParser
from Deeploy.Targets.Redmule.Tiler import RedmuleConvTilingReadyBindings, RedmuleGEMMTilingReadyBindings, \
    RedmuleMatMulTilingReadyBindings

MatMulRedmuleMapper = NodeMapper(MatMulParser(), RedmuleMatMulTilingReadyBindings)
Conv2DRedmuleMapper = NodeMapper(PULPFPConv2DParser(), RedmuleConvTilingReadyBindings)
GEMMMRedmuleMapper = NodeMapper(GEMMRedmuleParser(noBiasHoisting = False), RedmuleGEMMTilingReadyBindings)

RedmuleMapping = {
    'MatMul': MatMulLayer([MatMulRedmuleMapper]),
    # 'Conv' is intentionally not mapped here: the Redmule ConvTemplate
    # references the kernel symbol Conv2d_Im2Col_fp32_fp32_fp32_HWC_8_Redmule,
    # which is *declared* by the template but never *defined* in any source
    # file under TargetLibraries/.  Letting Conv fall through to the next
    # engine (PULPClusterEngine, which has a working
    # PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC implementation) keeps the
    # Siracusa+RedMulE link step from failing on undefined symbols.  When
    # a real RedMulE-accelerated Conv kernel lands, restore the mapping:
    #
    #     'Conv': ConvLayer([Conv2DRedmuleMapper]),
    'Gemm': GEMMLayer([GEMMMRedmuleMapper]),
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
