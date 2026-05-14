# ----------------------------------------------------------------------
#
# File: BasicParsers.py
#
# Last edited: 15.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import MatMulParser


class GEMMRedmuleParser(MatMulParser):

    def __init__(self, noBiasHoisting = True):
        # Order matters: super().__init__() of MatMulParser also writes
        # self.noBiasHoisting from its own default, so call super first and
        # then overwrite, otherwise our flag gets clobbered to True.
        super().__init__(noBiasHoisting = noBiasHoisting)
        self.noBiasHoisting = noBiasHoisting

    def parseNode(self, node: gs.Node) -> (bool):

        ret = all([len(node.inputs) >= 2, len(node.outputs) == 1, node.attrs['alpha'] == 1])

        if ret:
            if 'transA' in node.attrs:
                self.operatorRepresentation['transA'] = node.attrs['transA']
            else:
                self.operatorRepresentation['transA'] = 0

            if 'transB' in node.attrs:
                self.operatorRepresentation['transB'] = node.attrs['transB']
            else:
                self.operatorRepresentation['transB'] = 0
            if 'alpha' in node.attrs:
                self.operatorRepresentation['alpha'] = node.attrs['alpha']
            else:
                self.operatorRepresentation['alpha'] = 1
            if 'beta' in node.attrs:
                self.operatorRepresentation['beta'] = node.attrs['beta']
            else:
                self.operatorRepresentation['beta'] = 1

        return ret

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if ret:
            inputs = ['A', 'B']
            outputs = ['data_out']

            for idx, inputNode in enumerate(node.inputs):
                if idx < len(inputs):
                    self.operatorRepresentation[inputs[idx]] = newCtxt.lookup(inputNode.name).name
            for idx, outputNode in enumerate(node.outputs):
                self.operatorRepresentation[outputs[idx]] = newCtxt.lookup(outputNode.name).name

            if len(node.inputs) == 3:
                self.operatorRepresentation['C'] = newCtxt.lookup(node.inputs[2].name).name
            elif not self.noBiasHoisting:
                # Hoist a zero C tensor whose shape matches the GEMM output, so
                # the bias-required RedmuleGEMMTileConstraint and the existing
                # 3-operand kernel template can run unchanged on bias-less
                # Gemm nodes (e.g. backward GradFusedMatMul rewrites in CCT
                # training graphs that emit Y = A @ B with no C).
                outShape = node.outputs[0].shape
                values = np.zeros(outShape, dtype = np.float32)
                zeroTensor = gs.Constant(f'{node.name}_C_Tensor', values = values)
                newCtxt.hoistConstant(zeroTensor, _type = PointerClass(float32_t))
                # Also wire the hoisted Constant into the gs.Node inputs so the
                # tiler picks it up via its `node.inputs + node.outputs` walk,
                # AND register the Gemm as a user of the new buffer so the
                # MemoryConstraintFlow's kill-set analysis (which walks
                # `_users`) can find a consumer for it.  Without these the
                # tiler / flow analyzer KeyError or assert on the C tensor.
                node.inputs.append(zeroTensor)
                newCtxt.addUser(f'{node.name}_C_Tensor', node)
                self.operatorRepresentation['C'] = f'{node.name}_C_Tensor'

            self.operatorRepresentation['size'] = np.prod(newCtxt.lookup(node.inputs[0].name).shape)

        return newCtxt, ret
