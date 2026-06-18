# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, BlockingDmaFromAsyncDmaAdapter, DmaDirection, Future, \
    PerTensorWaitingStrategy


class L3DmaFuture(Future):

    _initTemplate = NodeTemplate("pi_cl_ram_req_t ${name} = {0};")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("")

    _waitTemplate = NodeTemplate("""
    if (${name}.size != 0) {
        pi_cl_ram_copy_wait(&${name});
        ${name}.size = 0;
    }""")


class L3Dma(AsyncDma):

    _transferTemplates = {
        2:
            NodeTemplate(
                "pi_cl_ram_copy_2d(get_ram_ptr(), ${ext}, ${loc}, ${transfer_size}, ${stride}, ${length}, ${ext2loc}, &${future});"
            )
    }
    _waitingStrategy = PerTensorWaitingStrategy(L3DmaFuture)

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        assert strideExt[-1] == 1, \
            "Mchan supports only contigous transfers of the innermost dimension for external memory"
        assert strideLoc[0] == shape[1] and strideLoc[1] == 1, \
            f"Mchan supports only contigous transfers for local memory. Received local shape: {shape}, stride: {strideLoc}"

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


# Default L3 DMA is async: it overlaps the next-tile L3<->L2 prefetch with
# compute, which is the whole point of DB and also speeds up SB hops. Almost
# every op is fine with this.
l3DmaHack = L3Dma()
# Per-op blocking variant for ops whose strided 2D pi_cl_ram_copy_2d trips the
# gvsoc hyper_v3 transfer_splitter tran_id leak under async (-> deadlock).
# Used ONLY by InPlaceAccumulatorV2 (accumulates conv-weight gradients, whose
# ConvGradW layout is strided). Routing just that op through the blocking
# adapter keeps every other op async/fast.
l3DmaBlocking = BlockingDmaFromAsyncDmaAdapter(L3Dma())
