# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, SubGraph, TransientBuffer
from Deeploy.TilingExtension.MemoryConstraints import PatternMemoryConstraints
from Deeploy.TilingExtension.MemoryScheduler import MemoryScheduler
from Deeploy.TilingExtension.TilerExtension import Tiler
from Deeploy.TilingExtension.TilerModel import TilerModel


def _isScalarBuffer(ctxt: NetworkContext, tensorName: str) -> bool:
    """A scalar tensor (product of dims == 1) cannot be split across two
    physical L1 slots; double-buffering it would just waste L1 and trip the
    DB pass's `_hoistMultibufferReferences` shape assertion. Treat as SB.
    """
    shape = ctxt.lookup(tensorName).shape
    if isinstance(shape, int):
        return shape <= 1
    return int(np.prod(shape)) <= 1


class DBOnlyL3Tiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        buffer = ctxt.lookup(tensorName)

        if isinstance(buffer, TransientBuffer):
            return 1

        if hop == 'L1':
            return 1

        return 2


class DBTiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        buffer = ctxt.lookup(tensorName)

        if isinstance(buffer, TransientBuffer):
            return 1

        if _isScalarBuffer(ctxt, tensorName):
            return 1

        return 2


class SBTiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        return 1


class TrainingMemoryScheduler(MemoryScheduler):
    """MemoryScheduler variant for training networks.

    Extends input tensor lifetimes to the end of the full tiling schedule so
    that forward-pass inputs remain live during the backward pass.
    """

    def _calculateLifetimes(self, ctxt: NetworkContext, patternMemoryConstraint: PatternMemoryConstraints,
                            memoryLevel: str) -> Tuple[Dict[str, Tuple[int, int]], Dict]:
        tensorLifetimeMap, tensorMap = super()._calculateLifetimes(ctxt, patternMemoryConstraint, memoryLevel)

        maxStepIdx = len(patternMemoryConstraint.nodeConstraints)
        for tensorName, lifetime in tensorLifetimeMap.items():
            buffer = ctxt.lookup(tensorName)
            if buffer.is_input:
                tensorLifetimeMap[tensorName] = (0, maxStepIdx)

        return tensorLifetimeMap, tensorMap


class TrainingSBTiler(SBTiler):
    memorySchedulerClass = TrainingMemoryScheduler


class TrainingDBTiler(DBTiler):
    memorySchedulerClass = TrainingMemoryScheduler

    # Operators where DB doesn't fit cleanly — fall back to SB for any pattern
    # containing one of these. Reasons:
    #   - SGD, InPlaceAccumulatorV2: in-place outputs aliased to inputs;
    #     DB's per-tensor multibuffer hoist would split the alias across two
    #     L1 slots and the in-place semantic breaks.
    #   - SoftmaxCrossEntropyLossGrad: produces output_grad that is consumed
    #     by *two* downstream Gemm nodes (multi-consumer intermediate); DB's
    #     hoist+egress logic interacts badly with MemoryAllocation's _live
    #     tracking and double-deallocates the tensor.
    DB_OPT_OUT_OPS = frozenset({
        "SGD",
        "InPlaceAccumulatorV2",
        # Loss + grad heads: small, with awkward shapes (multi-output, scalar,
        # or multi-consumer intermediates) — confuse DB hoist / dealloc.
        # DSCNN passes DB CI with SCE/SCEGrad opted out; MSE pair opted out
        # by analogy (autoencoder is the only model exercising them).
        "SoftmaxCrossEntropyLoss",
        "SoftmaxCrossEntropyLossGrad",
        "MSELoss",
        "MSELossGrad",
        # Gemm: backward Gemm under DB silently produces wrong gradients on
        # multi-tile training graphs (autoencoder DB CI: losses constant
        # ~0.097 across 4 update steps — model not learning — while DSCNN DB
        # Conv-only was numerically correct). Conservative opt-out until
        # backward Gemm DB egress is debugged. Conv DB still gives most of
        # the real cycle win on training graphs (DSCNN/MobileNet/ResNet).
        "Gemm",
    })

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        # If this pattern contains an opt-out op, force SB for every tensor in
        # this pattern. The DB pass then sees offsetList of length 1 and
        # returns applicable=False — SB.apply (run before DB.apply) has
        # already produced correct code for the pattern.
        for node in pattern:
            if node.op in self.DB_OPT_OUT_OPS:
                return 1
        return super().multiBufferStrategy(tilerModel, ctxt, pattern, path, hop, tensorName)
