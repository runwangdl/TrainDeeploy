# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future, PerTensorWaitingStrategy


class MchanTransferFuture(Future):
    _initTemplate = NodeTemplate("int ${name} = -1;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("${name} = mchan_transfer_get_id();")

    _waitTemplate = NodeTemplate("""
        if (${name} >= 0) {
            mchan_transfer_wait(${name});
            mchan_transfer_free(${name});
        }
        """)


class GAP9MchanDma(AsyncDma):

    _transferTemplates = {
        1:
            NodeTemplate(
                "{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext} }; mchan_transfer_push_1d(__mchan_tmp); }"
            ),
        2:
            NodeTemplate(
                "{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext}, .ext_size_1d = ${size_1d}, .ext_stride_1d = ${stride_2d} }; mchan_transfer_push_2d(__mchan_tmp); }"
            ),
    }
    # Chunked-1D fallback. `loc`/`ext` stay raw buffer identifiers so the dynamic
    # reference extractor in Closure.extractDynamicReferences can pick them up and
    # propagate them through closure args; byte offsets are applied in the C
    # expression instead of being baked into the identifier.
    _chunkedTransferTemplate = NodeTemplate("{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, "
                                            ".loc = (void *)((char *)${loc} + ${loc_offset}), "
                                            ".ext = (void *)((char *)${ext} + ${ext_offset}) }; "
                                            "mchan_transfer_push_1d(__mchan_tmp); }")

    # The mchan cmd size field is 17 bits, so it holds 0..(2^17 - 1) = 131071.
    # Using 1<<17 as the per-chunk max makes chunkSize carry into bit 17 and
    # clobber the direction/INC/event flags above it, producing a size-0 DMA whose
    # completion is scheduled at the current sim time -- which trips gvsoc's
    # "Time must be higher than current time". Same constant and same reason as
    # the PULPOpen backend.
    _MAX_1D_TRANSFER_BYTES = (1 << 17) - 1  # 131071 bytes

    # PerTensor, NOT Direction: GAP9 mchan allocates a fresh channel on every
    # descriptor enqueue (each mchan_transfer_push_* writes a new descriptor, and
    # the hardware advances to the next channel). DirectionWaitingStrategy shares
    # ONE future (one mchan_transfer_get_id) across all same-direction tensors of a
    # tile, so a tile with >1 input (e.g. SGD: weight + grad) emits one get_id but
    # multiple pushes -> the 2nd+ transfers run on channels that are never waited
    # nor freed -> mchan_transfer_wait() hangs forever (observed: OptimizerNetwork
    # SGD stalls at _sgd_blocks_5_pw_weight). PerTensor gives each tensor its own
    # get_id immediately before its push, matching mchan's 1 get_id : 1 push : 1
    # wait : 1 free contract.
    _waitingStrategy = PerTensorWaitingStrategy(MchanTransferFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        transferRank = len(shape)
        # MCHAN v7 requires contiguous transfers for innermost dimension in external memory
        assert strideExt[
            -1] == 1, "GAP9 MCHAN supports only contiguous transfers of the innermost dimension for external memory"

        # Local memory (TCDM) must also be contiguous
        if transferRank == 1:
            assert strideLoc[0] == 1, "GAP9 MCHAN supports only contiguous transfers for local memory"
        else:
            assert strideLoc[0] == shape[1] and strideLoc[
                1] == 1, "GAP9 MCHAN supports only contiguous transfers for local memory"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)

        # Build MCHAN command using flags from mchan.h
        # We construct the cmd value in Python and let the C code use the macros
        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        # <=, not a bit-count: ceil(log2(131072)) is 17 and would pass, yet 131072
        # does not fit a 17-bit field -- it carries into the flag bits.
        assert mchanTransferSize <= self._MAX_1D_TRANSFER_BYTES, (
            f"Transfer size {mchanTransferSize} exceeds the 17-bit mchan limit of "
            f"{self._MAX_1D_TRANSFER_BYTES} B. 1D transfers are chunked in transfer(); "
            "a 2D transfer this large needs the same treatment.")

        # cmd = (flags << 17) + size, matching PULPOpen MchanDma pattern
        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize
        operatorRepresentation["size"] = mchanTransferSize

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        """Split 1D transfers that exceed the 17-bit mchan size field.

        Tiled deployments never reach the limit because a tile is small, so this
        only fires for whole-tensor moves -- an untiled configuration, or a model
        whose tensor happens to exceed 128 KB. Without it those runs die at
        codegen with "transfer size is not representable with 17 bits" (ResNet-8
        147,456 B, CCT 262,144 B). The PULPOpen backend has had this since its
        own untiled work; GAP9 did not.
        """
        totalSize = math.prod(shape)
        if len(shape) == 1 and totalSize > self._MAX_1D_TRANSFER_BYTES:
            mchanFlags = 0
            mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0
            mchanFlags += (1 << 1)  # increment addresses
            mchanFlags += (1 << 3)  # event enable
            chunks: List[CodeSnippet] = []
            offset = 0
            while offset < totalSize:
                chunkSize = min(self._MAX_1D_TRANSFER_BYTES, totalSize - offset)
                opRepr: OperatorRepresentation = {
                    "loc": localBuffer.name,
                    "ext": externalBuffer.name,
                    "loc_offset": offset,
                    "ext_offset": offset,
                    "future": future.name,
                    "cmd": (mchanFlags << 17) + chunkSize,
                    "size": chunkSize,
                }
                chunks.append(CodeSnippet(self._chunkedTransferTemplate, opRepr))
                offset += chunkSize
            return chunks
        return super().transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)
