# ----------------------------------------------------------------------
#
# File: Platform.py
#
# Last edited: 08.05.2025
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
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

from Deeploy.DeeployTypes import TopologyOptimizer
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine, PULPConstantBuffer, PULPOptimizer, PULPPlatform, \
    PULPStructBuffer, PULPTransientBuffer, PULPVariableBuffer
from Deeploy.Targets.Redmule.Engine import PWConvGradW2DRedmuleMapper, PWConvGradX2DRedmuleMapper, RedmuleEngine

RedmuleOptimizer = TopologyOptimizer([*PULPOptimizer.passes])


class RedmulePlatform(PULPPlatform):

    def __init__(self,
                 engines = [RedmuleEngine("Redmule"), PULPClusterEngine("PULPCluster")],
                 variableBuffer = PULPVariableBuffer,
                 constantBuffer = PULPConstantBuffer,
                 structBuffer = PULPStructBuffer,
                 transientBuffer = PULPTransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)

        # Insert the RedMulE PWConvGrad mappers at position 0 of the
        # PULPClusterEngine's ConvGradW / ConvGradX layer mapper lists.
        # See the comment on RedmuleMapping in Engine.py: we cannot route
        # those op types through RedmuleEngine itself without confusing the
        # tiler, so we mutate the (still-pure-PULP) PULPClusterEngine layer
        # in place.  Order matters: PW Redmule must come before PULP's PW
        # mapper, so 1x1 ConvGrads bind to the RedMulE kernel; non-PW
        # variants fall through to PULP's DW / regular mappers as before.
        pulp_cluster = next((e for e in self.engines if e.name == "PULPCluster"), None)
        if pulp_cluster is not None:
            # Both PWConvGradW and PWConvGradX RedMulE mappers are hooked up
            # to PULPCluster's existing layer mapper lists.  ConvGradW was
            # disabled temporarily in 68d1639 because its template sized the
            # transpose buffer at C_in * H_in * W_in, which over-counted the
            # actual footprint for stride > 1 1x1 convs (ResNet8 layer2/3
            # downsample) and tripped tiler infeasibility on the regular-Conv
            # backward pattern memory.  After dropping that to the exact
            # C_in * H_out * W_out and teaching the kernel to sample X at
            # strided positions, the W path is back in.
            for op_type, redmule_mapper in (
                ("ConvGradW", PWConvGradW2DRedmuleMapper),
                ("ConvGradX", PWConvGradX2DRedmuleMapper),
            ):
                layer_factory = pulp_cluster.Mapping.get(op_type)
                if layer_factory is not None and hasattr(layer_factory, "maps"):
                    # Avoid double-inserting across repeated platform inits.
                    if redmule_mapper not in layer_factory.maps:
                        layer_factory.maps.insert(0, redmule_mapper)
