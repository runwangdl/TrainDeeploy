# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, OperatorRepresentation, VariableBuffer, \
    _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import ProfilingPrototypeMixIn, \
    PrototypeTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme


def _dropLeadingSingletons(shape, rank):
    """Drop leading size-1 dims of `shape` down to `rank`, when that is all that differs.

    Buffers can be registered with a leading batch-like singleton, e.g. a bias gradient as
    (1, 128), while the tile rectangles use the tensor's own rank, (128,). Comparing the two
    ranks directly misreads a sliced tensor as a full one.
    """
    shape = tuple(shape)
    extra = len(shape) - rank
    if extra > 0 and all(d == 1 for d in shape[:extra]):
        return shape[extra:]
    return shape


class SingleBufferingTilingCodeGeneration(TilingCodeGeneration):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 1)

    def _promotedByteOffsets(self, tensorName: str, bufferName: str, rects: List[HyperRectangle],
                             buf_shape: Tuple[int, ...], strides: Tuple[int, ...], typeWidth: int, combined_ends: list,
                             direction: DmaDirection) -> List[int]:
        """Byte offset of every tile into a promoted (standalone) buffer.

        The tiles' own offsets are all zero here: they are relative to the outer
        tile, which for an unpromoted tensor is a per-tile staging buffer. A promoted
        tensor has no staging buffer, so the true position has to come from elsewhere.

        1. Outer schedule -- exact. apply() re-derives the outer level's schedule for
           this node. When every outer tile holds exactly one inner tile, the inner
           window IS the outer window, so the outer rectangle's offset is the answer,
           halo of overlapping conv windows and padding included.
        2. Reconstruction -- exact only for a plain partition. Positions are rebuilt
           as tile_index x first_tile_dims, computeTileHyperRectangles' non-overlapping
           layout. That silently misplaced overlapping windows: a depthwise 3x3 reading
           a promoted MobileNetV1 activation split {144, 56} B over a 192 B row got its
           second tile at 144 instead of 136, and the forward pass was wrong from the
           first mini-batch. Every tile's size is now checked against that layout, and
           a mismatch is a hard error instead of wrong code.
        """
        N = len(rects)
        buf_rank = len(buf_shape)
        if buf_rank == 0:
            # Scalar buffer (e.g. the loss). Must precede every [-buf_rank:] slice:
            # dims[-0:] is the WHOLE tuple, not an empty one.
            return [0] * N
        tile_dims = rects[0].dims[-buf_rank:]
        if len(tile_dims) < buf_rank or tuple(tile_dims) == tuple(buf_shape):
            # Full tensor used every outer iteration: all offsets zero.
            return [0] * N

        def flat(offset: Tuple[int, ...]) -> int:
            return sum(o * st for o, st in zip(offset[-buf_rank:], strides)) * typeWidth // 8

        ctx = getattr(self, "_outerTileContext", None)
        if ctx is not None:
            outerIn, outerOut, innerLengths = ctx
            outerSteps = outerIn if direction == "ExternalToLocal" else outerOut
            if (all(n == 1 for n in innerLengths) and len(outerSteps) == N
                    and all(tensorName in step for step in outerSteps)):
                outer = [step[tensorName] for step in outerSteps]
                if all(
                        len(o.offset) >= buf_rank and tuple(o.dims[-buf_rank:]) == tuple(r.dims[-buf_rank:])
                        for o, r in zip(outer, rects)):
                    return [flat(o.offset) for o in outer]

        if N == 1 or math.prod(tile_dims) == math.prod(buf_shape):
            # Without an outer schedule: a single tile of a standalone buffer sits at its start, and a tile
            # with as many elements as the buffer is the whole buffer even when the
            # rectangle carries the consuming node's view of it rather than the
            # declared shape: a LayerNorm mean declared (64, 1) is tiled as (1, 64).
            # Comparing dims position by position rejected exactly that case.
            return [0] * N

        tile_ends = [math.ceil(buf_shape[d] / tile_dims[d]) for d in range(buf_rank)]
        windows = []  # same enumeration order as computeTileHyperRectangles
        idx = [0] * buf_rank
        for _ in range(math.prod(tile_ends)):
            windows.append(list(idx))
            for d in range(buf_rank):
                if idx[d] + 1 < tile_ends[d]:
                    idx[d] += 1
                    break
                idx[d] = 0
        M = len(windows)
        nontrivial = [d for d in range(buf_rank) if tile_ends[d] > 1]
        if len(nontrivial) == 1 and combined_ends:
            period_before = math.prod(combined_ends[:nontrivial[0]])
            sel = [(i // period_before) % M for i in range(N)]
        elif N <= M:
            sel = [i % M for i in range(N)]
        elif M * M > N:
            sel = [i // (N // M) for i in range(N)]
        else:
            sel = [i % M for i in range(N)]

        offsets = []
        for i, w in enumerate(sel):
            tidx = windows[w]
            expected = tuple(min(tile_dims[d], buf_shape[d] - tidx[d] * tile_dims[d]) for d in range(buf_rank))
            actual = tuple(rects[i].dims[-buf_rank:])
            if actual != expected:
                raise RuntimeError(
                    f"Promoted tensor '{bufferName}' ({tensorName}): tile {i} has dims {actual}, but the offset "
                    f"reconstruction assumes a non-overlapping layout giving {expected} (buffer {tuple(buf_shape)}, "
                    f"first tile {tuple(tile_dims)}), so the generated offsets would be wrong. Typically a kernel "
                    f"with a halo reads this promoted tensor across several inner tiles per outer tile. "
                    f"Exclude it from promotion, e.g. DEEPLOY_PROMOTE_SKIP_RE='^{bufferName}$'.")
            offsets.append(sum(tidx[d] * tile_dims[d] * strides[d] for d in range(buf_rank)) * typeWidth // 8)
        return offsets

    def _generateTransferScheduleCalls(
            self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
            transferSchedule: List[Dict[str, HyperRectangle]], tensorMemoryConstraintDict: Dict[str,
                                                                                                TensorMemoryConstraint],
            tileIdxVar: str, direction: DmaDirection) -> Tuple[NetworkContext, List[CodeSnippet], Set[Future]]:
        callStack: List[CodeSnippet] = []
        futures: Set[Future] = set()

        # Pre-scan: compute combined outer-loop tile counts across all tensors so
        # each Scenario-B tensor can determine its period_before (how many outer
        # iterations elapse before its own tiling dimension advances) without
        # relying on the sqrt heuristic, which fails when M_fast == M_slow.
        import math as _math
        _combined_ends: list = []
        for _tName, _rects_list in dictOfArrays(transferSchedule).items():
            try:
                _lBuf = ctxt.lookup(operatorRepresentation[_tName])
            except Exception:
                continue
            if not isinstance(_lBuf, _ReferenceBuffer):
                continue
            _eBuf = ctxt.lookup(_lBuf._referenceName)
            if isinstance(_eBuf, _ReferenceBuffer):
                continue
            _rects0 = list(_rects_list)
            if not _rects0:
                continue
            _bshape = _dropLeadingSingletons(_eBuf.shape, len(_rects0[0].dims))
            _br = len(_bshape)
            _td = _rects0[0].dims[-_br:]
            if len(_td) < _br:
                continue
            _te = [_math.ceil(_bshape[d] / _td[d]) for d in range(_br)]
            if not _combined_ends:
                _combined_ends = list(_te)
            else:
                for d in range(min(_br, len(_combined_ends))):
                    _combined_ends[d] = max(_combined_ends[d], _te[d])

        for tensorName, rectangles in dictOfArrays(transferSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = tensorMemoryConstraintDict[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            original_rectangles = list(rectangles)
            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            if externalBuffer._memoryLevel == self.externalMemory:
                typeWidth = localBuffer._type.referencedType.typeWidth
                buf_shape = _dropLeadingSingletons(externalBuffer.shape, len(original_rectangles[0].dims))
                buf_rank = len(buf_shape)
                _strides = [1] * buf_rank
                for _i, _d in enumerate(reversed(buf_shape[1:])):
                    _strides[_i + 1] = _strides[_i] * _d
                strides = tuple(reversed(_strides))
                all_zero = all(all(x == 0 for x in r.offset[-buf_rank:]) for r in original_rectangles)
                if all_zero and not isinstance(externalBuffer, _ReferenceBuffer):
                    cum_byte_offsets = self._promotedByteOffsets(tensorName, externalBuffer.name, original_rectangles,
                                                                 buf_shape, strides, typeWidth, _combined_ends,
                                                                 direction)
                elif all_zero:
                    # _ReferenceBuffer: outer UPDATE VARIABLE already advances
                    # this pointer; inner tiles use relative (0,...) offsets.
                    cum_byte_offsets = [0] * len(original_rectangles)
                else:
                    cum_byte_offsets = [
                        sum(o * s
                            for o, s in zip(r.offset[-buf_rank:], strides)) * typeWidth // 8
                        for r in original_rectangles
                    ]
                cum_buf = self._hoistValues(ctxt, f'{tensorName}_cumByteOffset', cum_byte_offsets)
                offset_expr = (f"{ctxt._mangle(cum_buf.name)}"
                               f"[*{ctxt._mangle(operatorRepresentation['tileIdxPtr'])}]")
                externalBufferRef = self._hoistReference(ctxt,
                                                         externalBuffer.name + "_ref",
                                                         externalBuffer,
                                                         shape = externalBufferShape,
                                                         offset = offset_expr,
                                                         override_type = VoidType)
            elif externalBuffer._memoryLevel == self.localMemory:
                # Buffer already resides at the local memory level (e.g. promoted to L2
                # while this is the L3→L2 pass). No DMA is needed at this level, but we
                # still need correct cumByteOffsets so the inner tiling pass advances
                # through the promoted buffer on each outer iteration.
                typeWidth = localBuffer._type.referencedType.typeWidth
                buf_shape = _dropLeadingSingletons(externalBuffer.shape, len(original_rectangles[0].dims))
                buf_rank = len(buf_shape)
                _strides = [1] * buf_rank
                for _i, _d in enumerate(reversed(buf_shape[1:])):
                    _strides[_i + 1] = _strides[_i] * _d
                strides = tuple(reversed(_strides))
                all_zero = all(all(x == 0 for x in r.offset[-buf_rank:]) for r in original_rectangles)
                if all_zero and not isinstance(externalBuffer, _ReferenceBuffer):
                    cum_byte_offsets = self._promotedByteOffsets(tensorName, externalBuffer.name, original_rectangles,
                                                                 buf_shape, strides, typeWidth, _combined_ends,
                                                                 direction)
                elif all_zero:
                    cum_byte_offsets = [0] * len(original_rectangles)
                else:
                    cum_byte_offsets = [
                        sum(o * s
                            for o, s in zip(r.offset[-buf_rank:], strides)) * typeWidth // 8
                        for r in original_rectangles
                    ]
                cum_buf = self._hoistValues(ctxt, f'{tensorName}_cumByteOffset', cum_byte_offsets)
                offset_expr = (f"{ctxt._mangle(cum_buf.name)}"
                               f"[*{ctxt._mangle(operatorRepresentation['tileIdxPtr'])}]")
                externalBufferRef = self._hoistReference(ctxt,
                                                         externalBuffer.name + "_ref",
                                                         externalBuffer,
                                                         shape = externalBufferShape,
                                                         offset = offset_expr,
                                                         override_type = VoidType)
                # Skip DMA generation; the inner level will handle L2→L1 transfers.
                continue
            else:
                externalBufferRef = self._hoistReference(ctxt,
                                                         externalBuffer.name + "_ref",
                                                         externalBuffer,
                                                         shape = externalBufferShape,
                                                         override_type = VoidType)

            future = self.dma.getFuture(tensorName, direction)

            # Allocate a future for this transfer
            if future not in futures:
                callStack.append(future.alloc())

            try:
                callStack.extend(
                    self._generateDmaTransferCalls(ctxt, tensorName, rectangles, tileIdxVar, localBuffer,
                                                   externalBufferRef, direction, future))
            except AssertionError as e:
                raise AssertionError(f"{e} while generating DMA transfer for tensor '{tensorName}'") from e

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, tileIdxVar,
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                callStack.append(referenceUpdate)

            futures.add(future)

        return ctxt, callStack, futures

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        # Single Buffering Tiling Loop Strategy
        # ===================================
        # - 1) Initialize all futures
        # - 2) for TILING_I in numTiles:
        #   - 2.1) Input data transfer for current tile (see "4.2) Input Data Transfers")
        #   - 2.2) Process current tile
        #   - 2.3) Output data transfer for current tile (see "4.4) Output Data Transfers")
        # - 3) Deinitialize all futures

        # 2) for TILING_I in numTiles:
        openLoopStatements = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        # 2.2) Input data transfer for current tile
        ctxt, ingressDMAStatements, ingressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
            nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "ExternalToLocal")

        ingressDMAStatements = [CodeSnippet(self._lineComment, {"comment": "Transfer input tiles"})
                               ] + ingressDMAStatements
        ingressDMAStatements += [CodeSnippet(self._lineComment, {"comment": "Wait for input tiles"})]
        ingressDMAStatements += [future.wait() for future in ingressFutures]

        # 2.4) Output data transfer for current tile
        ctxt, egressDMAStatements, egressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "LocalToExternal")
        egressDMAStatements = [CodeSnippet(self._lineComment, {"comment": "Transfer output tiles"})
                              ] + egressDMAStatements
        egressDMAStatements += [CodeSnippet(self._lineComment, {"comment": "Wait for output tiles"})]
        egressDMAStatements += [future.wait() for future in egressFutures]

        # 1) Initialize all futures
        setupStatements = [CodeSnippet(self._lineComment, {"comment": "Initialize DMA futures"})]
        setupStatements.extend([f.init() for f in ingressFutures | egressFutures])

        # 3) Deinitialize all futures
        teardownStatements = [CodeSnippet(self._lineComment, {"comment": "Deinitialize DMA futures"})]
        teardownStatements.extend([f.deinit() for f in ingressFutures | egressFutures])

        closeLoopStatements = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = operatorRepresentation['numTiles'],
                                  totalNumTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxPtr = operatorRepresentation['tileIdxPtr'],
                                  tileIdxVar = "TILING_I",
                                  kernelLevelTiling = True)

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDMAStatements, egressDMAStatements,
                                                    openLoopStatements, closeLoopStatements, setupStatements,
                                                    teardownStatements)

        return ctxt, executionBlock, True


class ProfilingSingleBufferingTilingMixIn(PrototypeTilingMixIn, ProfilingPrototypeMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: List[CodeSnippet],
                                     teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)

        executionBlock = cls.measurementArrayDeclaration(executionBlock, metaInfo, bufferingStr = "SB")

        executionBlock = cls.injectPrintCycleDiff(executionBlock, metaInfo)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: List[CodeSnippet], ingressDMAStatements: List[CodeSnippet],
                         egressDMAStatements: List[CodeSnippet],
                         closeLoopStatements: List[CodeSnippet]) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        _openLoopStatements = [openLoopStatements[0]]
        _openLoopStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _openLoopStatements += openLoopStatements[1:]

        _ingressDMAStatements = []
        _ingressDMAStatements += ingressDMAStatements
        _ingressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = cls.kernelProfilingWrap(executionBlock, metaInfo)

        _egressDMAStatements = []
        _egressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _egressDMAStatements += egressDMAStatements
        _egressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = super().generateLoopCode(executionBlock, metaInfo, _openLoopStatements, _ingressDMAStatements,
                                                  _egressDMAStatements, closeLoopStatements)
        return executionBlock
