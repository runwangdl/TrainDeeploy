# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, BlockingDmaFromAsyncDmaAdapter, DmaDirection, Future, \
    PerTensorWaitingStrategy


class GAP9L3DmaFuture(Future):

    _initTemplate = NodeTemplate("pi_cl_ram_req_t ${name} = {0};")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")

    _waitTemplate = NodeTemplate("""
    if (${name}.size != 0) {
        pi_cl_ram_copy_wait(&${name});
        ${name}.size = 0;
    }""")


class GAP9L3Dma(AsyncDma):

    _transferTemplates = {
        2:
            NodeTemplate(
                "pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t) ${ext}, ${loc}, ${transfer_size}, ${stride}, ${length}, ${ext2loc}, &${future});"
            )
    }
    _waitingStrategy = PerTensorWaitingStrategy(GAP9L3DmaFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideExt[-1] == 1, \
            "GAP9 RAM API requires contiguous transfers of the innermost dimension for external memory"
        assert strideLoc[0] == shape[1] and strideLoc[1] == 1, \
            f"GAP9 RAM API requires contiguous transfers for local memory. Received local shape: {shape}, stride: {strideLoc}"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)
        operatorRepresentation.update({
            "ext2loc": 1 if direction == "ExternalToLocal" else 0,
            "transfer_size": math.prod(shape),
            "length": shape[1],
            "stride": strideExt[0],
        })
        return operatorRepresentation


# --- Multi-request async L3 DMA (unlocks async DB for CHW / per-channel tiles) ---
# A CHW (NCHW) input tile is 3D; the Anydim adapter decomposes it into a per-channel
# loop of 2D copies. GAP9L3Dma tracks ONE pi_cl_ram_req_t per (tensor,direction), so
# under async DB the per-channel copies overwrite the single request handle and only
# the last is waited -> in-flight transfers + reused request -> UDMA overrun. This
# variant gives each per-channel copy its OWN request from a fixed static pool indexed
# by the Anydim loop variable; the wait drains the whole pool. Safe to async-overlap.
GAP9_L3DMA_POOL = 4096  # sanity cap; pools are sized exactly per-tile (see Anydim stamp)


class GAP9L3DmaMultiReqFuture(Future):
    # Pool is sized to the exact per-tile decomposition count (stamped by the Anydim
    # adapter as future._poolSize before f.init() is emitted); static => off CC-stack.

    _initTemplate = NodeTemplate("static pi_cl_ram_req_t ${name}_pool[${poolSize}] = {0};")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")

    _waitTemplate = NodeTemplate("""
    for (uint32_t _j = 0; _j < ${poolSize}; _j++) {
        if (${name}_pool[_j].size != 0) {
            pi_cl_ram_copy_wait(&${name}_pool[_j]);
            ${name}_pool[_j].size = 0;
        }
    }""")

    def _operatorRepresentation(self) -> OperatorRepresentation:
        return {"name": self.name, "poolSize": getattr(self, "_poolSize", 1)}


class GAP9L3DmaMultiReq(AsyncDma):

    _maxReq = GAP9_L3DMA_POOL  # Anydim asserts per-tile decomposition count <= this

    _transferTemplates = {
        2:
            NodeTemplate(
                "pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t) ${ext}, ${loc}, ${transfer_size}, ${stride}, ${length}, ${ext2loc}, &${future}_pool[${reqIdx}]);"
            )
    }
    _waitingStrategy = PerTensorWaitingStrategy(GAP9L3DmaMultiReqFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideExt[-1] == 1, \
            "GAP9 RAM API requires contiguous transfers of the innermost dimension for external memory"
        assert strideLoc[0] == shape[1] and strideLoc[1] == 1, \
            f"GAP9 RAM API requires contiguous transfers for local memory. Received local shape: {shape}, stride: {strideLoc}"

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)
        operatorRepresentation.update({
            "ext2loc": 1 if direction == "ExternalToLocal" else 0,
            "transfer_size": math.prod(shape),
            "length": shape[1],
            "stride": strideExt[0],
            "reqIdx": getattr(future, "_reqIndexExpr", "0"),
        })
        return operatorRepresentation


# Blocking adapter for L3 DMA (used in GAP9 L3 tiling)
gap9L3DmaHack = BlockingDmaFromAsyncDmaAdapter(GAP9L3Dma())
gap9L3DmaMultiReq = GAP9L3DmaMultiReq()
