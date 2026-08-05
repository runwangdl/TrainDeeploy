# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NodeMapper, Shape
from Deeploy.Targets.Generic.Layers import AddLayer, GEMMLayer, MulLayer, RQGEMMLayer, RQSConvLayer


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


class PULPMulLayer(MulLayer):
    """Mul that leaves a scalar second operand at its own shape.

    The generic layer rewrites the lower-rank operand to the higher-rank one's shape, so a
    single float becomes a tensor the size of the output. Nothing reads it: the float Mul
    kernel takes ``float32_t scalar = B[0]`` once and multiplies the whole tile by it, so
    the expansion is stored, transferred and then ignored.

    It is not a rounding error. LoRA's alpha/r scaling is one float, and expanding it once
    per element of the weight it scales costs 290.7 KB of ResNet8's 596.4 KB of L3
    constants and 510.5 KB of MobileNetV1's 1348.8 KB -- all of it holding the number 4.0.

    Only a genuine scalar is left alone. A vector operand still goes through the generic
    path, because the kernel would read only its first element and silently compute the
    wrong thing.
    """

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        big, small = (0, 1) if len(inputShapes[0]) >= len(inputShapes[1]) else (1, 0)
        if small == 1 and int(np.prod(inputShapes[small])) == 1 and len(inputShapes[big]) > 0:
            return (inputShapes, [inputShapes[big]])
        return super().computeShapes(inputShapes, outputShapes, operatorRepresentation, channels_first)


class PULPAddLayer(AddLayer):
    """Add that leaves a broadcast second operand at its own shape.

    The generic layer rewrites the lower-rank operand to the higher-rank one's
    shape, which turns a Linear's [O] bias into a full [1, M, O] tensor storing the
    same row M times. PULPBroadcastAddTileConstraint and PULPFloatAddTemplate
    handle the unequal shapes, so the rewrite is only needed when the operand is
    not a broadcast.
    """

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        big, small = (0, 1) if len(inputShapes[0]) >= len(inputShapes[1]) else (1, 0)
        if len(inputShapes[big]) != len(inputShapes[small]) and \
                len([d for d in inputShapes[small] if d != 1]) <= 1:
            return (inputShapes, [inputShapes[big]])
        return super().computeShapes(inputShapes, outputShapes, operatorRepresentation, channels_first)
