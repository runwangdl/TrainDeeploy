# ----------------------------------------------------------------------
#
# File: NeurekaBindings.py
#
# Last edited: 10.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# Luka Macan, University of Bologna
# Moritz Scherer, ETH Zurich
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

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker, GEMMChecker, MatMulChecker
from Deeploy.Targets.PULPOpen.Bindings import ClusterTransformer, ForkTransformer
from Deeploy.Targets.Redmule.Templates import ConvGradTemplate, ConvTemplate, GEMMTemplate, MatmulTemplate

RedmuleMatmulBindings = [
    NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                MatmulTemplate.referenceTemplate, ForkTransformer)
]

RedmuleConv2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), ConvTemplate.reference2DIm2ColTemplate,
        ForkTransformer)
]

RedmuleGEMMBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), GEMMTemplate.referenceTemplate,
        ForkTransformer)
]

# Pointwise (1x1) ConvGradW / ConvGradX routed to RedMulE.  The PULP versions
# (PULPFloatPWConvGradW2DBindings / PULPFloatPWConvGradX2DBindings) use the
# same ConvChecker signature, so the binding is identical apart from which
# template -> kernel symbol is selected.
RedmulePWConvGradW2DBindings = [
    NodeBinding(ConvChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ConvGradTemplate.referencePWConvGradW2DTemplate, ClusterTransformer)
]

RedmulePWConvGradX2DBindings = [
    NodeBinding(ConvChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ConvGradTemplate.referencePWConvGradX2DTemplate, ClusterTransformer)
]
