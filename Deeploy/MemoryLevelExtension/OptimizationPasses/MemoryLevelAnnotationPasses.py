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
            # Don't override a buffer that PromoteTensorsToL2 has already
            # moved to a non-default level. The annotation pipeline runs
            # multiple times (pre-bind, post-bind, codeTransform); on the
            # second/third invocation AnnotateIOMemoryLevel would reset
            # promoted graph I/O back to ioLevel (L3), but tiling codegen
            # from the first invocation already assumed L2 — the mismatch
            # causes InitTrainingNetwork to cl_ram_malloc (L3) while the
            # closure uses mchan (L1↔L2 only) → DMA hang.
            current = getattr(_buffer, '_memoryLevel', None)
            if current is not None and current != self.ioLevel:
                continue
            _buffer._memoryLevel = self.ioLevel

        return ctxt, graph


class PromoteTensorsToL2(SequentialPass):
    """Greedy L3→L2 tensor promotion with configurable selection strategies.

    Args:
        l2Size:                 L2 capacity in bytes (from MemoryHierarchy).
        headroom:               Bytes reserved for tile staging; not available for promotion.
                                Default 131072 covers the typical largest single tile-staging
                                buffer on 1 MB L2 configurations; raise it if minimalloc fails
                                with "capacity of N bytes" where N is too small for the actual
                                arena peak demand.
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

    # _SKIP_OPS: any buffer that is an input or output of a node with one of
    # these op types is excluded from promotion. Use this for ops whose
    # codegen has not been validated for L2-resident I/O.
    #
    # BatchNormInternal: training-mode BN has 3 outputs (data_out + saved_mean +
    # saved_inv_std). Deeploy's wrapTilingSolution asserts single output, so the
    # PULP tile constraint passes a single-output solution and grafts the
    # secondaries onto the schedule afterwards. That graft assumes the primary
    # has a per-tile data_out rectangle to slice the secondaries against, which
    # is only true while the primary's home is the default level (L3). Once
    # PromoteTensorsToL2 moves the primary to L2, the L2-target wrap sanitizes
    # data_out away (TilerExtension._annotateAddressSpaces deliberately leaves
    # the home-level addrSpace blank) and the workaround silently drops the
    # secondary -- the BN closure_L3 then never copies saved_mean / saved_inv
    # _std back to their L3 homes, BN backward reads stale L3 contents, and on
    # MobileNetV1 the simulated cluster deadlocks inside the BN backward fork
    # (verified via gvsoc --trace=insn ring trace: pe1-7 stuck at p.elw,
    # pe6/7 at the cluster_fork return-to-wait path for node_71).
    #
    # Excluding the whole BN node from promotion sidesteps that deadlock and
    # is the conservative fix until the multi-output tile path is rewritten.
    # NOTE: it does not, on its own, restore MobileNetV1 to 0/4 errors --
    # promote-with-BN-excluded still gives 3/4 errors with a small drift that
    # starts at the first backward pass, indicating a separate promote bug in
    # one of the non-BN op paths (Conv / ConvGrad / Transpose / ReluGrad).
    # That second bug is not in scope for this fix and needs its own
    # bisection.
    _SKIP_OPS = {
        'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity', 'BatchNormInternal', 'BatchNormalizationGrad',
        'LayerNormalization', 'LayerNormalizationGrad'
    }

    def __init__(self,
                 l2Size: int,
                 headroom: int = 131072,
                 strategy: str = 'cycle-aware',
                 includeActivations: bool = False,
                 maxBufferBytes: int = 2048,
                 minBufferBytes: int = 0,
                 setupCycles: int = 200,
                 bandwidthBytesPerCycle: float = 4.0,
                 seed: int = 42):
        super().__init__()
        self.l2Budget = l2Size - headroom
        self.strategy = strategy
        self.includeActivations = includeActivations
        self.maxBufferBytes = maxBufferBytes
        # minBufferBytes: reject candidates smaller than this. Tiny tensors
        # (BatchNorm scalars, bias vectors, etc.) each contribute negligible
        # cycle savings on their own but each promotion emits its own L2<->L1
        # staging block in the generated C code. On MobileNetV1 with cap=1MB
        # we saw 514 buffers promoted -> 135k lines of TrainingNetwork.c and
        # clang either OOMed or timed out in CI. A 4-8 KB floor keeps the
        # high-value medium/large buffers in the promotion pool while pruning
        # the long tail that bloats codegen without paying off in cycles.
        self.minBufferBytes = minBufferBytes
        self.setupCycles = setupCycles
        self.bw = bandwidthBytesPerCycle
        self.seed = seed

    def _bufferSize(self, buf: VariableBuffer) -> int:
        # buf.size_bytes() does not exist on any Deeploy buffer class -- the
        # original try/except always fell through to a hardcoded *4 multiplier
        # that assumed fp32 elements. That over-counts int8 pools (e.g. an
        # int8 PROMOTED_POOL_L2[163840] became 655 KB instead of 164 KB) and
        # under-counts fp16/fp64. Use the actual type width.
        try:
            return int(math.prod(buf.shape)) * buf._type.referencedType.typeWidth // 8
        except Exception:
            return math.prod(buf.shape) * 4

    @staticmethod
    def _sweepLinePeak(blocks):
        """Compute max simultaneous footprint of (size, lifetime) blocks.

        Pure-Python lower bound on minimalloc's actual packed peak; we use it
        in the greedy decision to decide if a candidate fits, leaving the
        actual offset assignment to the tile()-time pack pass that runs
        minimalloc for real.

        blocks: iterable of (size_bytes, (lower, upper)) tuples.
        Returns: int peak.
        """
        events = []
        for sz, lt in blocks:
            events.append((lt[0], +sz))
            events.append((lt[1] + 1, -sz))
        events.sort()
        peak = live = 0
        for _, delta in events:
            live += delta
            if live > peak:
                peak = live
        return peak

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        # If tile() has already frozen any buffer's allocation, this is the
        # post-tile call invoked from codeTransform. Promoting at this point
        # would change _memoryLevel on buffers whose codegen was emitted for
        # the old level, producing silently-corrupt output (see f8f1508).
        # Generalise the existing per-buffer guard to a global one: when any
        # buffer is frozen, accept zero new promotions for this whole call.
        any_frozen = any('allocTemplate' in b.__dict__ for b in {**ctxt.globalObjects, **ctxt.localObjects}.values())

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
            # Mirror the activation-branch f8f1508 guard: if the buffer's
            # tiling code has already been emitted (instance-level allocTemplate
            # pointing at MEMORYARENA_L3 + offset), flipping _memoryLevel to L2
            # post-hoc leaves the tiling closures writing to the wrong arena
            # and produces silently corrupt output. Skip those.
            if 'allocTemplate' in buf.__dict__:
                continue
            size = self._bufferSize(buf)
            if self.maxBufferBytes > 0 and size > self.maxBufferBytes:
                continue
            if size < self.minBufferBytes:
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
                if size < self.minBufferBytes:
                    continue
                candidates.append((name, buf, size, len(buf._users)))

            # Graph I/O lives in globalObjects as VariableBuffer (not ConstantBuffer).
            # Without this loop they stay in L3 even when there is plenty of L2 budget
            # left -- e.g. on ResNet8 training the unpromoted "input_*" / "output_*"
            # weight-and-grad tensors account for ~950 KB of L3 use. The Siracusa
            # training harness's l3_aware_copy() / IS_L2() helpers already handle
            # an L2-resident graph I/O destination correctly, so we can promote them.
            for name, buf in ctxt.globalObjects.items():
                if not isinstance(buf, VariableBuffer):
                    continue
                if isinstance(buf, (ConstantBuffer, _ReferenceBuffer)):
                    continue
                if isinstance(buf, TransientBuffer):
                    continue
                if not hasattr(buf, '_memoryLevel') or buf._memoryLevel != 'L3':
                    continue
                if name in skip_tensors:
                    continue
                if 'allocTemplate' in buf.__dict__:
                    continue
                # Skip single-use graph I/O (inference input_0 / output_0).
                # InitNetwork allocates these with cl_ram_malloc -> hyperram
                # regardless of _memoryLevel; the closure that follows assumes
                # the buffer is in L2 and uses mchan_transfer_1d (cluster idma)
                # which can only access L1/L2, not hyperram -> firmware polls
                # the DMA STATUS bit forever -> silent hang.
                # Training graph I/O (weights/grads) is multi-use and remains
                # eligible: len(_users) >= 2.
                if len(buf._users) <= 1:
                    continue
                size = self._bufferSize(buf)
                # Skip tiny graph I/O (step counters, reset flags, 1-element
                # scalars). These are training-loop control variables, not
                # weight tensors — promoting them wastes L2 and may confuse
                # the optimizer harness which shares pointers across networks.
                if size < 8:
                    continue
                if self.maxBufferBytes > 0 and size > self.maxBufferBytes:
                    continue
                if size < self.minBufferBytes:
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
            # Skip activations that have been packed into a shared pool;
            # the pool buffer itself contributes its packed_peak instead.
            if getattr(buf, '_packedIntoPool', None) is not None:
                return False
            # Skip buffers that have been frozen into an arena allocation by
            # tile() / _convertCtxtToStaticSchedule (instance-level allocTemplate
            # attribute). They live inside MEMORYARENA_LX at a fixed offset and
            # are accounted for via the arena, not as standalone bytes.
            if 'allocTemplate' in buf.__dict__:
                return False
            return getattr(buf, '_memoryLevel', None) == 'L2'

        already_l2 = sum(
            self._bufferSize(buf) for buf in {
                **ctxt.globalObjects,
                **ctxt.localObjects
            }.values() if _occupies_standalone_l2(buf))
        promoted = []
        # Refuse to promote anything once tile() has frozen allocations: the
        # codegen for each buffer was emitted for the level it had at tile time,
        # so flipping _memoryLevel here causes silent corruption (verified on
        # CCT_2_32_32_128: 6 post-tile promotions yielded 10/10 errors).
        if any_frozen:
            candidates = []

        # Lifetime-aware greedy: for activations whose _lifetime is set (which
        # happens when MemoryDeployerWrapper.bind() ran the pre-bind scheduler
        # walk), measure peak via sweep-line so non-overlapping activations can
        # share the same L2 bytes. Constants stay sum-counted -- they're
        # forever-alive read-only weights, no overlap possible.
        #
        # Tile staging guard: an activation whose untiled size exceeds the
        # smallest level (L1) would force tile() to L1-stage it whole and fail
        # the L1 minimalloc; reject those upfront.
        try:
            level_sizes = [getattr(self, '_memoryHierarchy', None)]
            level_sizes = []
            from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy as _MH  # noqa
        except Exception:
            pass
        l1_safety = float('inf')
        # The hierarchy isn't on `self`; pull it from ctxt indirectly.
        # If we have the hierarchy, reject activations whose untiled size
        # exceeds the smallest level (L1): an activation that doesn't fit
        # in L1 forces tile() to L1-stage it whole and the L1 minimalloc
        # fails. Constants are tiled by their consumer kernels and are
        # not subject to this constraint.
        for src_attr in ('memoryHierarchy', '_memoryHierarchy'):
            mh = getattr(ctxt, src_attr, None)
            if mh is not None and hasattr(mh, 'memoryLevels'):
                try:
                    l1_safety = min(lv.size for lv in mh.memoryLevels.values())
                    break
                except Exception:
                    pass

        const_used = 0  # bytes added to L2 by this call's const promotions
        var_blocks = []  # (size, lifetime) for ALL vars at L2 (prior + this call)
        # Seed var_blocks with already-promoted vars at L2 so sweep-line over
        # the union gives the right peak.
        for buf in ctxt.localObjects.values():
            if not isinstance(buf, VariableBuffer):
                continue
            if isinstance(buf, (ConstantBuffer, TransientBuffer, _ReferenceBuffer)):
                continue
            if getattr(buf, '_memoryLevel', None) != 'L2':
                continue
            if 'allocTemplate' in buf.__dict__:
                continue
            lt = getattr(buf, '_lifetime', None)
            if lt is None:
                continue
            var_blocks.append((self._bufferSize(buf), lt))
        # Fixed bytes occupied by stuff we cannot pack: prior consts + non-
        # lifetime-tracked prior buffers (already_l2 minus the var-sum we
        # replace with var-peak). Prior consts stay summed; prior vars get
        # replaced by their packed peak.
        prior_var_sum = sum(sz for sz, _ in var_blocks)
        fixed_prior = already_l2 - prior_var_sum  # consts + untracked

        def trial_total(extra_const_bytes, vblocks):
            return fixed_prior + const_used + extra_const_bytes + self._sweepLinePeak(vblocks)

        for name, buf, size, _ in candidates:
            is_const = isinstance(buf, ConstantBuffer) and not isinstance(buf, _ReferenceBuffer)
            if is_const:
                if trial_total(size, var_blocks) <= self.l2Budget:
                    buf._memoryLevel = 'L2'
                    const_used += size
                    promoted.append((name, size))
            else:
                if size >= l1_safety:
                    continue
                lt = getattr(buf, '_lifetime', None)
                if lt is None:
                    # No lifetime info -> can't measure overlap; charge full size
                    if trial_total(size, var_blocks) <= self.l2Budget:
                        buf._memoryLevel = 'L2'
                        const_used += size  # charge as if const (forever-alive)
                        promoted.append((name, size))
                    continue
                new_blocks = var_blocks + [(size, lt)]
                if trial_total(0, new_blocks) <= self.l2Budget:
                    buf._memoryLevel = 'L2'
                    var_blocks = new_blocks
                    promoted.append((name, size))

        final_var_peak = self._sweepLinePeak(var_blocks)
        l2_used = fixed_prior + const_used + final_var_peak

        print(f"  [PromoteTensorsToL2] promoted {len(promoted)} tensors, "
              f"{l2_used} / {self.l2Budget} bytes (already={already_l2}, "
              f"new_const={const_used}, var_peak={final_var_peak}, "
              f"strategy={self.strategy!r})")

        # Per-buffer dump for CI diagnostic. Lists each promoted buffer's
        # size, kind (const/var), lifetime window, and consumer node ops so we
        # can correlate a downstream codegen/sim failure back to a specific
        # buffer the pass chose.
        if promoted:
            # Build a name -> consumer-ops map from the graph.
            consumer_ops: dict = {}
            for node in graph.nodes:
                for t in list(node.inputs):
                    if t is not None:
                        consumer_ops.setdefault(t.name, []).append(node.op)
            print(f"  [PromoteTensorsToL2] dump (top {len(promoted)} by size):")
            for name, size in sorted(promoted, key = lambda x: -x[1]):
                buf = ctxt.lookup(name)
                kind = 'const' if isinstance(buf, ConstantBuffer) else 'var'
                lt = getattr(buf, '_lifetime', None)
                lt_s = f"[{lt[0]},{lt[1]}]" if lt else "-"
                ops = ','.join(sorted(set(consumer_ops.get(name, [])))) or '-'
                print(f"    {kind:<5} {size:>8} B  lt={lt_s:<10}  ops={ops}  name={name}")

        if l2_used > self.l2Budget:
            print(f"  [PromoteTensorsToL2] WARNING: standalone L2 footprint {l2_used} B "
                  f"exceeds promote budget {self.l2Budget} B by {l2_used - self.l2Budget} B. "
                  f"This usually means tile() hoisted additional buffers to L2 after the "
                  f"earlier promote calls; minimalloc may then place arena buffers in "
                  f"physically-occupied addresses and produce silently wrong output.")

        return ctxt, graph
