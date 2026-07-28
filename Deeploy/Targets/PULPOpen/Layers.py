# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NodeMapper, Shape
from Deeploy.Targets.Generic.Layers import GEMMLayer, RQGEMMLayer, RQSConvLayer


class PULPRQSConvLayer(RQSConvLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if channels_first:
            inputShapes[2] = [outputShapes[0][1]]  # Channels out dimension of Kernel
            inputShapes[3] = [outputShapes[0][1]]  # Channels out dimension of Kernel
        else:
            inputShapes[2] = [outputShapes[0][-1]]  # Channels out dimension of Kernel
            inputShapes[3] = [outputShapes[0][-1]]  # Channels out dimension of Kernel
        return (inputShapes, outputShapes)


class PULPRQSGEMMLayer(RQGEMMLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:

        if operatorRepresentation['transB']:
            channelDim = -2
        else:
            channelDim = -1

        inputShapes[2] = [inputShapes[1][channelDim]]  # Channels out dimension of Kernel
        inputShapes[3] = [inputShapes[1][channelDim]]  # Channels out dimension of Kernel

        return (inputShapes, outputShapes)


class PULPGEMMLayer(GEMMLayer):
    """GEMM that keeps a one-dimensional bias one-dimensional.

    The generic layer widens the C operand to [M, N] unconditionally. That is what
    the kernel used to require, and for a transformer it is expensive: every Linear
    bias is O values shared by all M rows, so the widened operand stores the same
    vector M times. On CCT-2 at 64 tokens each 128-wide bias occupies 32 KB to carry
    512 B, and the eight of them live at the peak account for 252 KB, around 13% of
    the measured peak under LoRA and linear probing.

    PULPFloatGEMMTemplate now passes the bias row stride to the kernel, so a
    one-dimensional C is read with stride 0 and needs no widening. A C operand that
    arrives with a real second dimension is left alone and behaves as before.
    """

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if len(inputShapes) == 3 and len([d for d in inputShapes[2] if d != 1]) <= 1:
            return (inputShapes, outputShapes)  # broadcast bias: leave as [O]
        return super().computeShapes(inputShapes, outputShapes, operatorRepresentation, channels_first)
