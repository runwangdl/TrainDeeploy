# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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


class SingleBufferingTilingCodeGeneration(TilingCodeGeneration):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 1)

    def _generateTransferScheduleCalls(
            self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
            transferSchedule: List[Dict[str, HyperRectangle]], tensorMemoryConstraintDict: Dict[str,
                                                                                                TensorMemoryConstraint],
            tileIdxVar: str, direction: DmaDirection,
            hoistable_tensors: Set[str] = None) -> Tuple[NetworkContext, List[CodeSnippet], Set[Future],
                                                          List[CodeSnippet], Set[Future]]:
        """Generate DMA transfer calls for all tensors in the schedule.

        Returns:
            (ctxt, perTileCallStack, perTileFutures, hoistedCallStack, hoistedFutures)
            hoistedCallStack contains DMA calls for tensors in hoistable_tensors
            (should be emitted once before the tile loop).
        """
        callStack: List[CodeSnippet] = []
        hoistedCallStack: List[CodeSnippet] = []
        futures: Set[Future] = set()
        hoistedFutures: Set[Future] = set()
        if hoistable_tensors is None:
            hoistable_tensors = set()

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
            _bshape = _eBuf.shape
            _br = len(_bshape)
            _rects0 = list(_rects_list)
            if not _rects0:
                continue
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
                buf_shape = externalBuffer.shape
                buf_rank = len(buf_shape)
                _strides = [1] * buf_rank
                for _i, _d in enumerate(reversed(buf_shape[1:])):
                    _strides[_i + 1] = _strides[_i] * _d
                strides = tuple(reversed(_strides))
                all_zero = all(all(x == 0 for x in r.offset[-buf_rank:]) for r in original_rectangles)
                if all_zero and not isinstance(externalBuffer, _ReferenceBuffer):
                    # Standalone promoted buffer: no outer-loop pointer advance.
                    # Detect whether the inner tiles span the full tensor (Scenario A:
                    # same full tensor every outer iteration) or are slices of it
                    # (Scenario B: tensor tiled across outer iterations).
                    # Use the first tile's dims to distinguish: if tile == full tensor,
                    # all cumulative offsets are 0; otherwise enumerate outer windows
                    # in column-major order (matching computeTileHyperRectangles).
                    tile_dims = original_rectangles[0].dims[-buf_rank:]
                    if len(tile_dims) < buf_rank or tuple(tile_dims) == tuple(buf_shape):
                        # Full tensor used every outer iteration — all offsets zero.
                        cum_byte_offsets = [0] * len(original_rectangles)
                    else:
                        import math as _math
                        tile_ends = [_math.ceil(buf_shape[d] / tile_dims[d]) for d in range(buf_rank)]

                        def _colmaj(ends):
                            idx = [0] * len(ends)
                            total = 1
                            for e in ends:
                                total *= e
                            for _ in range(total):
                                yield list(idx)
                                for d in range(len(ends)):
                                    if idx[d] + 1 < ends[d]:
                                        idx[d] += 1
                                        break
                                    else:
                                        idx[d] = 0

                        base_offsets = [
                            sum(ti * tile_dims[d] * strides[d]
                                for d, ti in enumerate(tidx)) * typeWidth // 8
                            for tidx in _colmaj(tile_ends)
                        ]
                        num_rects = len(original_rectangles)
                        M = len(base_offsets)
                        N = num_rects
                        nontrivial = [d for d in range(buf_rank) if tile_ends[d] > 1]
                        if len(nontrivial) == 1 and _combined_ends:
                            d_tile = nontrivial[0]
                            period_before = 1
                            for _d in range(d_tile):
                                period_before *= _combined_ends[_d]
                            cum_byte_offsets = [base_offsets[(i // period_before) % M] for i in range(N)]
                        elif N <= M:
                            # Fewer outer tiles than unique buffer windows: take one-to-one.
                            cum_byte_offsets = [base_offsets[i % M] for i in range(N)]
                        elif M * M > N:
                            cum_byte_offsets = [base_offsets[i // (N // M)] for i in range(N)]
                        else:
                            cum_byte_offsets = [base_offsets[i % M] for i in range(N)]
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
                buf_shape = externalBuffer.shape
                buf_rank = len(buf_shape)
                _strides = [1] * buf_rank
                for _i, _d in enumerate(reversed(buf_shape[1:])):
                    _strides[_i + 1] = _strides[_i] * _d
                strides = tuple(reversed(_strides))
                all_zero = all(all(x == 0 for x in r.offset[-buf_rank:]) for r in original_rectangles)
                if all_zero and not isinstance(externalBuffer, _ReferenceBuffer):
                    tile_dims = original_rectangles[0].dims[-buf_rank:]
                    if len(tile_dims) < buf_rank or tuple(tile_dims) == tuple(buf_shape):
                        cum_byte_offsets = [0] * len(original_rectangles)
                    else:
                        import math as _math
                        tile_ends = [_math.ceil(buf_shape[d] / tile_dims[d]) for d in range(buf_rank)]

                        def _colmaj(ends):
                            idx = [0] * len(ends)
                            total = 1
                            for e in ends:
                                total *= e
                            for _ in range(total):
                                yield list(idx)
                                for d in range(len(ends)):
                                    if idx[d] + 1 < ends[d]:
                                        idx[d] += 1
                                        break
                                    else:
                                        idx[d] = 0

                        base_offsets = [
                            sum(ti * tile_dims[d] * strides[d]
                                for d, ti in enumerate(tidx)) * typeWidth // 8
                            for tidx in _colmaj(tile_ends)
                        ]
                        num_rects = len(original_rectangles)
                        M = len(base_offsets)
                        N = num_rects
                        nontrivial = [d for d in range(buf_rank) if tile_ends[d] > 1]
                        if len(nontrivial) == 1 and _combined_ends:
                            d_tile = nontrivial[0]
                            period_before = 1
                            for _d in range(d_tile):
                                period_before *= _combined_ends[_d]
                            cum_byte_offsets = [base_offsets[(i // period_before) % M] for i in range(N)]
                        elif N <= M:
                            cum_byte_offsets = [base_offsets[i % M] for i in range(N)]
                        elif M * M > N:
                            cum_byte_offsets = [base_offsets[i // (N // M)] for i in range(N)]
                        else:
                            cum_byte_offsets = [base_offsets[i % M] for i in range(N)]
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

            # Route to hoisted or per-tile list
            is_hoisted = tensorName in hoistable_tensors
            target_stack = hoistedCallStack if is_hoisted else callStack
            target_futures = hoistedFutures if is_hoisted else futures

            # Allocate a future for this transfer
            if future not in target_futures:
                target_stack.append(future.alloc())

            if is_hoisted:
                # Generate a single (non-tiled) DMA for the first rectangle only
                try:
                    target_stack.extend(
                        self._generateDmaTransferCalls(ctxt, tensorName, rectangles[:1], tileIdxVar, localBuffer,
                                                       externalBufferRef, direction, future))
                except AssertionError as e:
                    raise AssertionError(f"{e} while generating hoisted DMA for tensor '{tensorName}'") from e
            else:
                try:
                    target_stack.extend(
                        self._generateDmaTransferCalls(ctxt, tensorName, rectangles, tileIdxVar, localBuffer,
                                                       externalBufferRef, direction, future))
                except AssertionError as e:
                    raise AssertionError(f"{e} while generating DMA transfer for tensor '{tensorName}'") from e

                referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, tileIdxVar,
                                                                        externalBufferRef)
                if referenceUpdate is not None:
                    target_stack.append(referenceUpdate)

            target_futures.add(future)

        return ctxt, callStack, futures, hoistedCallStack, hoistedFutures

    @staticmethod
    def _detectHoistableTensors(transferSchedule: List[Dict[str, HyperRectangle]]) -> Set[str]:
        """Detect tensors whose HyperRectangle is identical across all tile iterations.
        These can be loaded once before the tile loop instead of every iteration."""
        from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
        hoistable: Set[str] = set()
        if len(transferSchedule) <= 1:
            return hoistable
        for tensorName, rectangles in dictOfArrays(transferSchedule).items():
            rects = list(rectangles)
            if len(rects) <= 1:
                continue
            first = rects[0]
            if all(r.offset == first.offset and r.dims == first.dims for r in rects[1:]):
                hoistable.add(tensorName)
        return hoistable

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        # Single Buffering Tiling Loop Strategy
        # ===================================
        # - 0) Hoist DMA for tensors that don't change across tiles
        # - 1) Initialize all futures
        # - 2) for TILING_I in numTiles:
        #   - 2.1) Input data transfer for current tile (skip hoisted)
        #   - 2.2) Process current tile
        #   - 2.3) Output data transfer for current tile (skip hoisted)
        # - 3) Deinitialize all futures

        # 0) Detect hoistable input tensors (identical HyperRectangle across all tiles)
        hoistableIngress = self._detectHoistableTensors(tilingSchedule.inputLoadSchedule)

        # 2) for TILING_I in numTiles:
        openLoopStatements = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        # 2.2) Input data transfer — single call splits into per-tile + hoisted
        ctxt, ingressDMAStatements, ingressFutures, hoistedIngressDMA, hoistedIngressFutures = \
            self._generateTransferScheduleCalls(
                ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
                nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "ExternalToLocal",
                hoistable_tensors=hoistableIngress)

        ingressDMAStatements = [CodeSnippet(self._lineComment, {"comment": "Transfer input tiles"})
                               ] + ingressDMAStatements
        ingressDMAStatements += [CodeSnippet(self._lineComment, {"comment": "Wait for input tiles"})]
        ingressDMAStatements += [future.wait() for future in ingressFutures]

        # 2.4) Output data transfer for current tile
        ctxt, egressDMAStatements, egressFutures, _, _ = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "LocalToExternal")
        egressDMAStatements = [CodeSnippet(self._lineComment, {"comment": "Transfer output tiles"})
                              ] + egressDMAStatements
        egressDMAStatements += [CodeSnippet(self._lineComment, {"comment": "Wait for output tiles"})]
        egressDMAStatements += [future.wait() for future in egressFutures]

        # 1) Initialize all futures (including hoisted)
        allFutures = ingressFutures | egressFutures | hoistedIngressFutures
        setupStatements = [CodeSnippet(self._lineComment, {"comment": "Initialize DMA futures"})]
        setupStatements.extend([f.init() for f in allFutures])

        # Add hoisted DMA transfers before the loop (loaded once, reused across all tiles)
        if hoistedIngressDMA:
            setupStatements += [CodeSnippet(self._lineComment, {"comment": "Hoisted DMA: tensors unchanged across tiles"})]
            setupStatements += hoistedIngressDMA
            setupStatements += [future.wait() for future in hoistedIngressFutures]

        # 3) Deinitialize all futures
        teardownStatements = [CodeSnippet(self._lineComment, {"comment": "Deinitialize DMA futures"})]
        teardownStatements.extend([f.deinit() for f in allFutures])

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
