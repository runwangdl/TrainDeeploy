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

    # Operators where DB cannot fall through the scalar-pattern check below.
    # All other "previously opted out" ops (SoftmaxCrossEntropyLoss, MSELoss,
    # MSELossGrad, Gemm) are now handled by the scalar-pattern check — they
    # all have a scalar tensor (loss, lazy_reset_grad) somewhere in their
    # pattern that triggers the fall-back to SB.
    DB_OPT_OUT_OPS = frozenset({
        # In-place alias outputs (output is _alias'd to an input). DB's
        # per-tensor multibuffer hoist would split the alias across two L1
        # slots and break in-place semantics.
        "SGD",
        # SoftmaxCrossEntropyLossGrad's output_grad is consumed by 2 backward
        # Gemms (multi-consumer intermediate) — DB's per-consumer hoist
        # inflates _users and breaks MemoryAllocation _live tracking.
        "SoftmaxCrossEntropyLossGrad",
        # Transpose (forward) and ConvGradW (backward weight-gradient) are the
        # two ops whose MULTI-TILE data rearrangement is computed wrong under L3
        # double-buffering — found by an SB-vs-DB per-op checksum node-diff on
        # MobileNetV1 (forward step-0 loss diverged at the first conv block; the
        # backward error compounded over steps). Both reorder data across tiles
        # (Transpose: NHWC<->NCHW; ConvGradW: im2col-style correlation) and the
        # per-tile DB transfer of the remainder/edge tile reads stale data.
        # SINGLE-TILE cases are correct (the bug is multi-tile only):
        #   - ResNet8 (CCF, small single-tile convs, no NHWC transpose): passes
        #     either way, a hair slower with ConvGradW SB (74->77M/step).
        #   - CCT (contiguous transposes): unaffected (104M/step, was 103M).
        #   - MobileNetV1 (large multi-tile activations + NHWC input transpose):
        #     these two opt-outs are what make its DB pass (was 4/4 wrong).
        # ConvGradX double-buffers correctly and is deliberately NOT opted out.
        # TODO: real fix is in the multi-tile DB rearrange codegen; this
        # per-op-type opt-out is the safe, validated stop-gap.
        "Transpose",
        "ConvGradW",
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
        # If ANY tensor in this pattern is scalar (product-of-dims <= 1),
        # force coefficient=1 for the WHOLE pattern. Otherwise we end up
        # with mixed coefficients (scalar=1, non-scalar=2) — neither
        # SB.apply (needs all=1) nor DB.apply (needs all=2) is applicable
        # and the codegen degenerates to a bare kernel call with NO DMA
        # setup, so the kernel reads stale L1 data. This was the real
        # cause of the "autoencoder weights frozen" symptom previously
        # mis-attributed to Gemm: MSELoss's scalar `loss` output triggered
        # this degenerate case.
        #
        # Exception: InPlaceAccumulatorV2's lazy_reset_grad is a control
        # flag pinned full by TileConstraint — it won't trigger mixed
        # coefficients because its multiBufferCoefficient is forced to 1
        # via _isScalarBuffer in the base Tiler class.
        for node in pattern:
            is_inplace_acc = node.op == "InPlaceAccumulatorV2"
            for tensor in list(node.inputs) + list(node.outputs):
                tname = tensor.name
                if ctxt.is_buffer(tname) and _isScalarBuffer(ctxt, tname):
                    if is_inplace_acc and "lazy_reset_grad" in tname:
                        continue  # Skip: pinned scalar flag, not a data tensor
                    return 1
        return super().multiBufferStrategy(tilerModel, ctxt, pattern, path, hop, tensorName)


class TrainingDBOnlyL3Tiler(TrainingDBTiler):
    """L3-mode training DB: double-buffer only the L3→L2 hop, leave L2→L1 SB.

    Mirrors the inference path's `DBOnlyL3Tiler`. Plain `TrainingDBTiler` doubles
    every memory hop's coefficient — for `defaultMemLevel=L3` that means L2
    staging buffers also get doubled, which blows the 2 MB L2 capacity on
    ResNet8/MobileNetV1 training graphs (constraint solver returns infeasible).
    """

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        if hop == "L1":
            return 1
        # InPlaceAccumulatorV2: force coefficient=1 (SB) for the WHOLE pattern
        # so it takes the blocking L3 DMA path. It is an in-place op (acc +=
        # grad on one L3 address), and for conv-weight-grad accumulators the
        # gradient is strided 2D (from ConvGradW tiling). Double-buffering it
        # issues an *async* strided pi_cl_ram_copy_2d, which trips the gvsoc
        # UDMA hyper_v3 transfer_splitter tran_id leak -> pi_cl_ram_copy_wait
        # never returns -> ResNet8/MobileNetV1 deadlock. Blocking (SB) waits
        # each transfer inline, so the strided copy completes before the next
        # is issued. CCT's accumulators are contiguous so it is unaffected
        # either way; its DB win comes from the (contiguous) Transpose hops,
        # which still double-buffer via the async dbDma.
        for node in pattern:
            if node.op == "InPlaceAccumulatorV2":
                return 1
        result = super().multiBufferStrategy(tilerModel, ctxt, pattern, path, hop, tensorName)
        return result
