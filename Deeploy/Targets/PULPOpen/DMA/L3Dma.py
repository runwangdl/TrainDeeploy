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


# async L3 DMA — used ONLY as the DB (dbDma) backend, where it overlaps the
# next-tile L3<->L2 prefetch with compute (the source of the DB speedup). It is
# deliberately NOT used for SB hops: SB waits on each tile before computing, so
# async buys no overlap, only the strided-2D deferred-wait corruption/deadlock
# risk. (Matches the PULPL3Tiling docstring and the GAP9 binding.)
l3DmaHack = L3Dma()
# blocking L3 DMA — the SB-hop backend (dma=). Waits each transfer inline so
# strided 2D ConvGrad transfers complete in order: no corruption, no UDMA
# deadlock. Matches devel's behaviour (devel used blocking for every hop).
l3DmaBlocking = BlockingDmaFromAsyncDmaAdapter(L3Dma())
