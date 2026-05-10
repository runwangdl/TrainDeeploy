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
            # Only ConvGradX is hooked up.  PWConvGradW2DRedmuleMapper exists
            # (kernel + template + binding all in tree) but inserting it into
            # the PULPCluster ConvGradWLayer makes the tiler infeasible on
            # ResNet8 / MobileNetV1 -- the pattern-memory solver appears to
            # account for the inserted mapper's transient buffer footprint
            # even on 3x3 / DW ConvGradW nodes where the parser declines and
            # the mapper is never selected (see exit=1 reproducer with
            # only-W-inserted in /tmp/v6_r8.log, while only-X-inserted is
            # green).  Diagnosing properly needs a deeper look at how
            # TilingReadyNodeBindings instances aggregate across layer
            # mappers; left as a follow-up.
            #
            # ConvGradX has the same construction but doesn't trip it,
            # presumably because its template's transposeBuffer footprint
            # (C_in * C_out) is identical in size to PULP's existing PW
            # template, while ConvGradW introduces a new C_in*H_in*W_in
            # buffer that PULP's regular ConvGradW kernel never hoists.
            for op_type, redmule_mapper in (
                ("ConvGradX", PWConvGradX2DRedmuleMapper),
            ):
                layer_factory = pulp_cluster.Mapping.get(op_type)
                if layer_factory is not None and hasattr(layer_factory, "maps"):
                    # Avoid double-inserting across repeated platform inits.
                    if redmule_mapper not in layer_factory.maps:
                        layer_factory.maps.insert(0, redmule_mapper)
