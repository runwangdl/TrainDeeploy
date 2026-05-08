# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import NetworkContext, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy


class AnnotateDefaultMemoryLevel(SequentialPass):

    def __init__(self, memoryHierarchy: MemoryHierarchy):
        super().__init__()
        self.memoryHierarchy = memoryHierarchy

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _buffer in {**ctxt.localObjects, **ctxt.globalObjects}.values():
            if not hasattr(_buffer, "_memoryLevel"):
                _buffer._memoryLevel = self.memoryHierarchy.getDefaultMemoryLevel().name
        return ctxt, graph


class AnnotateIOMemoryLevel(SequentialPass):

    def __init__(self, ioLevel: str):
        super().__init__()
        self.ioLevel = ioLevel

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        buffers = []

        def globalBuffers(tensors: List[gs.Tensor]) -> List[VariableBuffer]:
            return [ctxt.globalObjects[tensor.name] for tensor in tensors if tensor.name in ctxt.globalObjects.keys()]

        inputBuffers = globalBuffers(graph.inputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer) and len(_buffer._users) > 0,
                          inputBuffers)

        outputBuffers = globalBuffers(graph.outputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer), outputBuffers)

        for _buffer in buffers:
            _buffer._memoryLevel = self.ioLevel

        return ctxt, graph


class AnnotateTrainingWeightPromotionToL2(SequentialPass):
    """Override _memoryLevel to L2 for weight graph-inputs in a training graph.

    Training graph inputs are ordered:
        [0 .. num_data_inputs)   — data inputs  (keep at current level, e.g. L3)
        [num_data_inputs .. end) — weights + grad-acc buffers

    Among the latter, any input whose name contains ``_grad.accumulation.buffer``
    is a gradient-accumulation buffer (already assigned to L2 by the harness);
    the remainder are trainable weights and are promoted here to L2 so that they
    are allocated in on-chip SRAM (pi_l2_malloc) rather than HyperRAM
    (cl_ram_malloc), eliminating per-step L3 DMA for those tensors.

    Must be inserted into the pass list *after* AnnotateIOMemoryLevel so that
    the initial L3 annotation is already in place when this pass runs.
    """

    _GRAD_ACC = "_grad.accumulation.buffer"

    def __init__(self, num_data_inputs: int):
        super().__init__()
        self.num_data_inputs = num_data_inputs

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for i, tensor in enumerate(graph.inputs):
            if i < self.num_data_inputs:
                continue
            if self._GRAD_ACC in tensor.name:
                continue
            buf = ctxt.globalObjects.get(tensor.name)
            if buf is not None and isinstance(buf, ctxt.VariableBuffer):
                buf._memoryLevel = "L2"
        return ctxt, graph