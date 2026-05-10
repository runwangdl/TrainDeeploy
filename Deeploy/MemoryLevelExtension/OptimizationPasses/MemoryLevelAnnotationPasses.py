# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
import random as _random
from typing import List, Tuple

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, TransientBuffer, VariableBuffer, _ReferenceBuffer
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


class PromoteTensorsToL2(SequentialPass):
    """Greedy L3→L2 tensor promotion with configurable selection strategies.

    Args:
        l2Size:                 L2 capacity in bytes (from MemoryHierarchy).
        headroom:               Bytes reserved for tile staging; not available for promotion.
        strategy:               One of 'cycle-aware' (default), 'greedy-score',
                                'knapsack-ratio', 'smallest', 'largest', 'random'.
        includeActivations:     If True, also promote VariableBuffers (activations) from
                                localObjects, not just ConstantBuffers from globalObjects.
        maxBufferBytes:         Skip buffers larger than this; 0 means no cap.
        setupCycles:            Per-DMA-transaction fixed overhead in cycles (default 200).
        bandwidthBytesPerCycle: Effective L3↔L2 bandwidth (default 4.0 B/cycle).
        seed:                   Random seed for the 'random' strategy (default 42).

    Excludes tensors that are inputs or outputs of SkipTransformer ops (Reshape, Squeeze,
    Unsqueeze, Flatten, Identity). Promoting one side of such a pointer-alias pair without
    the other causes a cross-level alias that crashes at runtime.
    """

    _SKIP_OPS = {'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity'}

    def __init__(self,
                 l2Size: int,
                 headroom: int = 64000,
                 strategy: str = 'cycle-aware',
                 includeActivations: bool = False,
                 maxBufferBytes: int = 2048,
                 setupCycles: int = 200,
                 bandwidthBytesPerCycle: float = 4.0,
                 seed: int = 42):
        super().__init__()
        self.l2Budget = l2Size - headroom
        self.strategy = strategy
        self.includeActivations = includeActivations
        self.maxBufferBytes = maxBufferBytes
        self.setupCycles = setupCycles
        self.bw = bandwidthBytesPerCycle
        self.seed = seed

    def _bufferSize(self, buf: VariableBuffer) -> int:
        try:
            return buf.size_bytes()
        except Exception:
            return math.prod(buf.shape) * 4

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        skip_tensors: set = set()
        for node in graph.nodes:
            if node.op in self._SKIP_OPS:
                for t in list(node.inputs) + list(node.outputs):
                    if t is not None:
                        skip_tensors.add(t.name)

        candidates: List[Tuple[str, VariableBuffer, int, int]] = []

        for name, buf in ctxt.globalObjects.items():
            if not isinstance(buf, ConstantBuffer) or isinstance(buf, _ReferenceBuffer):
                continue
            if not hasattr(buf, '_memoryLevel') or buf._memoryLevel != 'L3':
                continue
            if name in skip_tensors:
                continue
            size = self._bufferSize(buf)
            if self.maxBufferBytes > 0 and size > self.maxBufferBytes:
                continue
            candidates.append((name, buf, size, len(buf._users)))

        if self.includeActivations:
            for name, buf in ctxt.localObjects.items():
                if isinstance(buf, _ReferenceBuffer) or isinstance(buf, ConstantBuffer):
                    continue
                if isinstance(buf, TransientBuffer):
                    continue
                if not hasattr(buf, '_memoryLevel') or buf._memoryLevel != 'L3':
                    continue
                if name in skip_tensors:
                    continue
                # Skip buffers whose tiling code was already generated with L3 semantics.
                # _convertCtxtToStaticSchedule sets an instance-level allocTemplate that
                # points to MEMORYARENA_L3 + offset.  Promoting such a buffer to L2 after
                # tile() has run leaves the tiling closures writing to the L2 arena scratch
                # buffer instead of the actual named L2 buffer, producing corrupt output.
                if 'allocTemplate' in buf.__dict__:
                    continue
                size = self._bufferSize(buf)
                if self.maxBufferBytes > 0 and size > self.maxBufferBytes:
                    continue
                candidates.append((name, buf, size, len(buf._users)))

        if self.strategy == 'cycle-aware':
            candidates.sort(key = lambda x: x[3] * (self.setupCycles + x[2] / self.bw) / max(x[2], 1), reverse = True)
        elif self.strategy == 'greedy-score':
            candidates.sort(key = lambda x: x[3] * x[2], reverse = True)
        elif self.strategy == 'knapsack-ratio':
            candidates.sort(key = lambda x: x[3], reverse = True)
        elif self.strategy == 'smallest':
            candidates.sort(key = lambda x: x[2])
        elif self.strategy == 'largest':
            candidates.sort(key = lambda x: x[2], reverse = True)
        elif self.strategy == 'random':
            _random.Random(self.seed).shuffle(candidates)
        else:
            raise ValueError(f"Unknown promotion strategy: {self.strategy!r}")

        # Account for tensors already promoted to L2 by a previous call to this pass.
        # MemoryDeployerWrapper calls apply() up to 3 times; each call must not exceed
        # the shared L2 budget, so we subtract what is already committed.
        #
        # Exclude buffers that don't physically occupy standalone L2 storage:
        #   - _ReferenceBuffer: aliases another buffer, no bytes of its own
        #   - TransientBuffer:  scratch, managed via the tiling arena, not standalone
        #   - "MEMORYARENA" in name: allocator-internal arena scratch
        # Without these guards, post-tile invocations double-count tile-staging buffers
        # that happen to be tagged at L2, inflating already_l2 by hundreds of KB and
        # making the "promoted N / M bytes" log line meaningless.
        def _occupies_standalone_l2(buf) -> bool:
            if isinstance(buf, _ReferenceBuffer):
                return False
            if isinstance(buf, TransientBuffer):
                return False
            if "MEMORYARENA" in buf.name:
                return False
            return getattr(buf, '_memoryLevel', None) == 'L2'

        already_l2 = sum(
            self._bufferSize(buf)
            for buf in {**ctxt.globalObjects, **ctxt.localObjects}.values()
            if _occupies_standalone_l2(buf)
        )
        l2_used = already_l2
        promoted = []
        for name, buf, size, _ in candidates:
            if l2_used + size <= self.l2Budget:
                buf._memoryLevel = 'L2'
                l2_used += size
                promoted.append((name, size))

        print(f"  [PromoteTensorsToL2] promoted {len(promoted)} tensors, "
              f"{l2_used} / {self.l2Budget} bytes (already={already_l2}, new={l2_used - already_l2}, "
              f"strategy={self.strategy!r})")

        if l2_used > self.l2Budget:
            print(f"  [PromoteTensorsToL2] WARNING: standalone L2 footprint {l2_used} B "
                  f"exceeds promote budget {self.l2Budget} B by {l2_used - self.l2Budget} B. "
                  f"This usually means tile() hoisted additional buffers to L2 after the "
                  f"earlier promote calls; minimalloc may then place arena buffers in "
                  f"physically-occupied addresses and produce silently wrong output.")

        return ctxt, graph
