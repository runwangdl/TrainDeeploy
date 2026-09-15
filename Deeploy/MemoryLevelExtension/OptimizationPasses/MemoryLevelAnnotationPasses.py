# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
import random as _random
from typing import Dict, List, Optional, Tuple

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
    # The pointer-aliasing subset of _SKIP_OPS: output = view of input[0].
    _ALIAS_OPS = {'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Identity'}

    class _AliasGroup:
        """One promotion candidate standing for a closure of buffers that share storage."""

        def __init__(self, name: str, members: List[VariableBuffer], lifetime, flat: bool):
            self.name = name
            self.members = members
            self._lifetime = lifetime
            self._flat = flat

        @property
        def _memoryLevel(self):
            return self.members[0]._memoryLevel

        @_memoryLevel.setter
        def _memoryLevel(self, level):
            for m in self.members:
                m._memoryLevel = level

    def __init__(self,
                 l2Size: int,
                 headroom: int = 131072,
                 strategy: str = 'cycle-aware',
                 includeActivations: bool = False,
                 maxBufferBytes: int = 2048,
                 minBufferBytes: int = 0,
                 setupCycles: int = 200,
                 bandwidthBytesPerCycle: float = 4.0,
                 seed: int = 42,
                 fetchBytes: Optional[Dict[str, int]] = None):
        super().__init__()
        # Measured L3<->L2 traffic per tensor, harvested from a tiling solution in
        # which nothing was promoted -- Q(t) is the traffic saved by promoting t, so it
        # can only be read off a run where t is still in L3.
        #
        # Without it the access count is len(buf._users), the number of consuming
        # nodes, which counts at SCHEDULE granularity only. The L3->L2 DMA sits inside
        # the per-tile loop bounded by TILING_CODEGEN_L2_<node>_numTiles, so a tensor an
        # operator re-reads on every tile is fetched once per tile rather than once.
        # Measured on ResNet8, 32 of 92 nodes have a trip count above 1 and one reaches
        # 20; per tensor the correction runs to 4.9x on ResNet8, 25x on MobileNetV1 and
        # 64.5x on CCT, and it changes the selected set on the latter two.
        self.fetchBytes = fetchBytes or {}
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

    def _accesses(self, name: str, size: int, buf) -> float:
        """How many times the tensor's own size crosses the level boundary.

        With measured traffic available this is traffic/size, so the score keeps
        its units. A tensor the harvest did not see does NOT fall back to the
        consuming-node count: the harvest covers every tiled pattern it could
        establish a trip count for (it reports noTileCount, which is 0 on all
        five training graphs), so absence means the tensor never crosses the
        boundary inside the tile loop. Falling back there mixes bytes with node
        counts and lets tensors that save nothing outrank tensors that save real
        traffic -- which is what made the ratio strategies pick candidates whose
        contribution to the reported traffic removal is exactly zero.
        """
        if self.fetchBytes:
            measured = self.fetchBytes.get(name)
            if not measured or size <= 0:
                return 0.0
            return measured / size
        return len(getattr(buf, '_users', []) or [])

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

        # DEEPLOY_PROMOTE_SKIP_OPS_KEEP narrows _SKIP_OPS to the listed ops, so the
        # exclusion can be attributed per op type instead of all-or-nothing. The set
        # mixes two unrelated hazards: pointer-aliasing ops (Reshape/Squeeze/...) and
        # multi-output ops whose tiling graft assumes an L3 home (BatchNormInternal,
        # LayerNormalization and their grads).
        import os as _os1
        _keep = _os1.environ.get('DEEPLOY_PROMOTE_SKIP_OPS_KEEP')
        _skipOps = set(_keep.split(',')) if _keep else set(self._SKIP_OPS)

        # Alias groups (DEEPLOY_PROMOTE_ALIAS_GROUPS=1). Reshape / Squeeze / Unsqueeze /
        # Flatten / Identity emit `out = in`: the output is a view of the input's storage
        # (the PULP ReshapeTemplate records it as out._alias = in.name at bind time).
        # Promoting one side without the other hands a HyperRAM offset to the cluster DMA
        # as if it were an L2 address (CCT-2 with the filter off: "Got error during
        # transfer (addr: 0x1ff3e98)"). With grouping on, the whole alias closure is ONE
        # candidate -- one size, the union of the lifetimes, the sum of the measured
        # traffic -- and _memoryLevel is set on every member together; the alias ops then
        # leave the skip set (the multi-output BatchNorm / LayerNorm entries stay unless
        # DEEPLOY_PROMOTE_SKIP_OPS_KEEP says otherwise). The pool packer in
        # TilerExtension gives a view the slot of its root, so a group costs its size once.
        _groupsOn = bool(_os1.environ.get('DEEPLOY_PROMOTE_ALIAS_GROUPS'))
        _parent: Dict[str, str] = {}

        def _find(x: str) -> str:
            while _parent.get(x, x) != x:
                _parent[x] = _parent.get(_parent[x], _parent[x])
                x = _parent[x]
            return x

        def _union(a: str, b: str):
            ra, rb = _find(a), _find(b)
            _parent.setdefault(a, a)
            _parent.setdefault(b, b)
            if ra != rb:
                _parent[rb] = ra

        if _groupsOn:
            if not _keep:
                _skipOps -= self._ALIAS_OPS
            for node in graph.nodes:
                if node.op in self._ALIAS_OPS and node.inputs and node.outputs \
                        and node.inputs[0] is not None and node.outputs[0] is not None:
                    _union(node.inputs[0].name, node.outputs[0].name)
            for _nm, _bf in list(ctxt.globalObjects.items()) + list(ctxt.localObjects.items()):
                _al = getattr(_bf, '_alias', None)
                if _al:
                    _union(_nm, _al)

        skip_tensors: set = set()
        for node in graph.nodes:
            if node.op in _skipOps:
                for t in list(node.inputs) + list(node.outputs):
                    if t is not None:
                        skip_tensors.add(t.name)

        # Rejection census: why each L3 buffer never became a promotion candidate.
        # Enabled with DEEPLOY_PROMOTE_CENSUS=1. The point is to see what actually
        # caps L2 occupancy -- the budget, or the eligibility filters.
        import collections as _coll
        import os as _os0
        _censusOn = bool(_os0.environ.get('DEEPLOY_PROMOTE_CENSUS'))
        _censusN = _coll.Counter()
        _censusB = _coll.Counter()

        def _rej(reason, sz = 0):
            if _censusOn:
                _censusN[reason] += 1
                _censusB[reason] += sz

        candidates: List[Tuple[str, VariableBuffer, int, int]] = []

        for name, buf in ctxt.globalObjects.items():
            if not isinstance(buf, ConstantBuffer) or isinstance(buf, _ReferenceBuffer):
                continue
            if not hasattr(buf, '_memoryLevel') or buf._memoryLevel != 'L3':
                continue
            if name in skip_tensors and not _os0.environ.get('DEEPLOY_PROMOTE_IGNORE_SKIP_OPS'):
                _rej('SKIP_OPS neighbour', self._bufferSize(buf))
                continue
            # Mirror the activation-branch f8f1508 guard: if the buffer's
            # tiling code has already been emitted (instance-level allocTemplate
            # pointing at MEMORYARENA_L3 + offset), flipping _memoryLevel to L2
            # post-hoc leaves the tiling closures writing to the wrong arena
            # and produces silently corrupt output. Skip those.
            if 'allocTemplate' in buf.__dict__:
                _rej('frozen allocTemplate', self._bufferSize(buf))
                continue
            size = self._bufferSize(buf)
            if self.maxBufferBytes > 0 and size > self.maxBufferBytes:
                continue
            if size < self.minBufferBytes:
                continue
            accesses = self._accesses(name, size, buf)
            candidates.append((name, buf, size, accesses))

        if self.includeActivations:
            for name, buf in ctxt.localObjects.items():
                if isinstance(buf, _ReferenceBuffer) or isinstance(buf, ConstantBuffer):
                    continue
                if isinstance(buf, TransientBuffer):
                    _rej('transient scratch', self._bufferSize(buf))
                    continue
                if not hasattr(buf, '_memoryLevel') or buf._memoryLevel != 'L3':
                    continue
                if name in skip_tensors and not _os0.environ.get('DEEPLOY_PROMOTE_IGNORE_SKIP_OPS'):
                    _rej('SKIP_OPS neighbour', self._bufferSize(buf))
                    continue
                # Skip buffers whose tiling code was already generated with L3 semantics.
                # _convertCtxtToStaticSchedule sets an instance-level allocTemplate that
                # points to MEMORYARENA_L3 + offset.  Promoting such a buffer to L2 after
                # tile() has run leaves the tiling closures writing to the L2 arena scratch
                # buffer instead of the actual named L2 buffer, producing corrupt output.
                if 'allocTemplate' in buf.__dict__:
                    _rej('frozen allocTemplate', self._bufferSize(buf))
                    continue
                size = self._bufferSize(buf)
                if self.maxBufferBytes > 0 and size > self.maxBufferBytes:
                    continue
                if size < self.minBufferBytes:
                    continue
                accesses = self._accesses(name, size, buf)
                candidates.append((name, buf, size, accesses))

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
                    _rej('transient scratch', self._bufferSize(buf))
                    continue
                if not hasattr(buf, '_memoryLevel') or buf._memoryLevel != 'L3':
                    continue
                if name in skip_tensors and not _os0.environ.get('DEEPLOY_PROMOTE_IGNORE_SKIP_OPS'):
                    _rej('SKIP_OPS neighbour', self._bufferSize(buf))
                    continue
                if 'allocTemplate' in buf.__dict__:
                    _rej('frozen allocTemplate', self._bufferSize(buf))
                    continue
                # Skip single-use graph I/O (inference input_0 / output_0).
                # InitNetwork allocates these with cl_ram_malloc -> hyperram
                # regardless of _memoryLevel; the closure that follows assumes
                # the buffer is in L2 and uses mchan_transfer_1d (cluster idma)
                # which can only access L1/L2, not hyperram -> firmware polls
                # the DMA STATUS bit forever -> silent hang.
                # Training graph I/O (weights/grads) is multi-use and remains
                # eligible: len(_users) >= 2.
                if len(buf._users) <= 1 and not _os0.environ.get('DEEPLOY_PROMOTE_ALLOW_SINGLE_USE_IO'):
                    _rej('graph I/O with <=1 user', self._bufferSize(buf))
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
                accesses = self._accesses(name, size, buf)
                candidates.append((name, buf, size, accesses))

        if _groupsOn:
            # Merge the candidates of one alias closure into one candidate. Every buffer of
            # the closure that exists in the context must itself be a candidate -- otherwise
            # the storage would end up split across levels, which is the crash we avoid.
            _known = set(_parent)
            _inCtxt = lambda n: n in ctxt.globalObjects or n in ctxt.localObjects
            _byRoot: Dict[str, list] = {}
            for c in candidates:
                _byRoot.setdefault(_find(c[0]), []).append(c)
            _closure: Dict[str, set] = {}
            for n in _known:
                if _inCtxt(n):
                    _closure.setdefault(_find(n), set()).add(n)
            _merged = []
            for root, members in _byRoot.items():
                names = {m[0] for m in members}
                need = _closure.get(root, names)
                if len(members) == 1 and len(need) <= 1:
                    _merged.append(members[0])
                    continue
                if names != need:
                    for m in members:
                        _rej('alias group partly ineligible', m[2])
                    continue
                size = max(m[2] for m in members)
                Q = sum(m[3] * m[2] for m in members)
                lts = [getattr(m[1], '_lifetime', None) for m in members]
                lt = None if any(l is None for l in lts) else (min(l[0] for l in lts), max(l[1] for l in lts))
                flat = any((isinstance(m[1], ConstantBuffer) and not isinstance(m[1], _ReferenceBuffer))
                           or m[0] in ctxt.globalObjects for m in members)
                rootName = ctxt.dealiasBuffer(members[0][0]) if hasattr(members[0][1], '_alias') else root
                if rootName not in names:
                    rootName = members[0][0]
                _merged.append((rootName, self._AliasGroup(rootName, [m[1] for m in members], lt, flat), size, Q / size))
            candidates = _merged

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
        elif self.strategy == 'traffic-per-peak':
            pass  # ordering is decided adaptively below, not by a static sort
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
        # Graph I/O promoted to L2 is allocated standalone for the whole program --
        # InitNetwork emits one pi_l2_malloc per tensor, and they are NOT packed into
        # PROMOTED_POOL_L2 the way activations are. Charging them by sweep-line overlap
        # therefore lets several of them claim the same bytes in the budget while the
        # generated code asks the allocator for all of them at once. On CCT training that
        # understated L2 by 256 KB (four 64 KB weights) and the runtime allocator failed
        # while initialising the optimizer network. They are weights and gradients: live
        # from the first step to the last, so there is no overlap to exploit anyway.
        graphIONames = {
            name for name, buf in ctxt.globalObjects.items()
            if isinstance(buf, VariableBuffer) and not isinstance(buf, (ConstantBuffer, _ReferenceBuffer,
                                                                        TransientBuffer))
        }

        def _isStandaloneGraphIO(buf) -> bool:
            return getattr(buf, 'name', None) in graphIONames

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

        # Split already_l2 into const (sum) and var (sweep-line peak).
        # Before tile() runs, _packedIntoPool isn't set, so promoted vars
        # are still standalone. Using raw sum vastly overestimates the L2
        # footprint (~4MB on MobileNetV1 vs ~783KB packed peak), causing
        # the budget check to reject valid promotions and minimalloc to
        # overlap MEMORYARENA_L2 with the promotion pool.
        _already_const = 0
        _already_var_blocks = []
        for buf in {**ctxt.globalObjects, **ctxt.localObjects}.values():
            if not _occupies_standalone_l2(buf):
                continue
            sz = self._bufferSize(buf)
            lt = getattr(buf, '_lifetime', None)
            if lt is not None and not isinstance(buf, ConstantBuffer) and not _isStandaloneGraphIO(buf):
                _already_var_blocks.append((sz, lt))
            else:
                _already_const += sz
        already_l2 = _already_const + self._sweepLinePeak(_already_var_blocks)
        # Experiment hooks (measurement only, read from the environment so they
        # need no plumbing through three runners). ONLY/SKIP take a regex over the
        # buffer name; NO_L1_SAFETY lifts the "activation larger than L1" rejection.
        import os as _os
        import re as _re
        _onlyRe = _os.environ.get('DEEPLOY_PROMOTE_ONLY_RE')
        _skipRe = _os.environ.get('DEEPLOY_PROMOTE_SKIP_RE')
        _onlyRe = _re.compile(_onlyRe) if _onlyRe else None
        _skipRe = _re.compile(_skipRe) if _skipRe else None
        if _onlyRe is not None:
            candidates = [c for c in candidates if _onlyRe.search(c[0])]
        if _skipRe is not None:
            candidates = [c for c in candidates if not _skipRe.search(c[0])]

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

        if _os.environ.get('DEEPLOY_PROMOTE_NO_L1_SAFETY'):
            # Historical limitation: activations whose untiled size exceeds L1 were
            # rejected upfront. Lifting it is the point of the experiment.
            l1_safety = float('inf')

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
                _rej('frozen allocTemplate', self._bufferSize(buf))
                continue
            lt = getattr(buf, '_lifetime', None)
            if lt is None or _isStandaloneGraphIO(buf):
                continue
            var_blocks.append((self._bufferSize(buf), lt))
        # Fixed bytes: the const portion of already_l2. The var portion is
        # tracked via var_blocks (sweep-line), so fixed_prior must NOT
        # subtract the raw var sum (already_l2 uses sweep peak for vars).
        fixed_prior = _already_const

        def trial_total(extra_const_bytes, vblocks):
            return fixed_prior + const_used + extra_const_bytes + self._sweepLinePeak(vblocks)

        def _isFlat(buf, lt):
            """True if the candidate must be charged its full size at every step."""
            return (isinstance(buf, ConstantBuffer) and not isinstance(buf, _ReferenceBuffer)) \
                or lt is None or _isStandaloneGraphIO(buf) or getattr(buf, '_flat', False)

        if self.strategy == 'traffic-per-peak':
            # Residency as the paper states it: repeatedly admit
            #     t* = argmax_t  Q(t) / max(Delta_P(t), eps),
            # with Q(t) the off-chip traffic keeping t on-chip removes and
            # Delta_P(t) = Lambda(P + t) - Lambda(P) the increase of the peak of the
            # live resident bytes. Every admission reshapes the live profile, so all
            # scores are recomputed before the next choice -- that is what separates
            # this from a knapsack with fixed per-item costs, and what the older
            # single-sort strategies do not do.
            #
            # Delta is read off a per-step live profile instead of re-running the
            # sweep line for every (round, candidate): adding s over steps [lo, hi]
            # lifts that window to max(profile[lo:hi+1]) + s, so
            #     Delta = max(0, max(profile[lo:hi+1]) + s - peak).
            #
            # The ratio greedy alone is myopic, and measurably so: on the 640-wide
            # MLperf autoencoder it admits small high-ratio tensors first, fragments
            # the budget and stops at 82.6% of L2 having removed 47.4% of the
            # traffic, while a plain size-descending fill reaches 97.2% and 51.6%.
            # So we also build the descending-Q solution and keep whichever removes
            # more traffic -- both are greedy passes over the same candidate set and
            # the objective is measured, not estimated.
            import numpy as _np
            _EPS = 1.0

            _pool = []
            for name, buf, size, acc in candidates:
                lt = getattr(buf, '_lifetime', None)
                if not _isFlat(buf, lt) and size >= l1_safety:
                    continue
                _pool.append((name, buf, size, acc, lt))

            _T = 1 + max((lt[1] for _, lt in var_blocks), default = 0)
            for _n, _b, _sz, _acc, _lt in _pool:
                if _lt is not None:
                    _T = max(_T, _lt[1] + 1)

            def _plan(adaptive):
                """Return (chosen, savedQ, blocks, constBytes) without mutating state."""
                profile = _np.zeros(_T, dtype = _np.int64)
                for sz, (lo, hi) in var_blocks:
                    profile[lo:hi + 1] += sz
                peak = int(profile.max()) if _T else 0
                constUsed = 0
                blocks = list(var_blocks)
                chosen = []
                savedQ = 0
                remaining = list(_pool)
                if not adaptive:
                    # Fixed order: most traffic first, ties to the larger tensor.
                    remaining.sort(key = lambda c: (c[3] * c[2], c[2]), reverse = True)
                while remaining:
                    curTotal = fixed_prior + constUsed + peak
                    best = None
                    bestScore = 0.0
                    bestDelta = 0
                    for cand in remaining:
                        nm, bf, sz, acc, lt = cand
                        Q = acc * sz
                        if Q <= 0:
                            continue  # promotion stops paying once nothing saves traffic
                        if _isFlat(bf, lt):
                            delta = sz
                            newTotal = curTotal + sz
                        else:
                            localMax = int(profile[lt[0]:lt[1] + 1].max())
                            newPeak = max(peak, localMax + sz)
                            delta = newPeak - peak
                            newTotal = fixed_prior + constUsed + newPeak
                        if newTotal > self.l2Budget:
                            continue  # admitted only if the resulting peak still fits
                        if adaptive:
                            score = Q / max(delta, _EPS)
                            if best is None or score > bestScore:
                                best, bestScore, bestDelta = cand, score, delta
                        else:
                            best, bestDelta = cand, delta
                            break
                    if best is None:
                        break
                    nm, bf, sz, acc, lt = best
                    if _isFlat(bf, lt):
                        constUsed += sz
                    else:
                        profile[lt[0]:lt[1] + 1] += sz
                        peak += bestDelta
                        blocks.append((sz, lt))
                    chosen.append((nm, bf, sz))
                    savedQ += acc * sz
                    remaining.remove(best)
                return chosen, savedQ, blocks, constUsed

            _ratio = _plan(adaptive = True)
            _dense = _plan(adaptive = False)
            _chosen, _savedQ, _blocks, _constUsed = _ratio if _ratio[1] >= _dense[1] else _dense
            if _ratio[1] != _dense[1]:
                print(f"  [PromoteTensorsToL2] traffic-per-peak: ratio-greedy saves "
                      f"{_ratio[1]:,} B, descending-Q saves {_dense[1]:,} B -- keeping "
                      f"{'ratio-greedy' if _ratio[1] >= _dense[1] else 'descending-Q'}")
            for _nm, _bf, _sz in _chosen:
                _bf._memoryLevel = 'L2'
                if isinstance(_bf, self._AliasGroup):
                    for _m in _bf.members:
                        promoted.append((_m.name, self._bufferSize(_m)))
                else:
                    promoted.append((_nm, _sz))
            var_blocks = _blocks
            const_used = _constUsed
        else:
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
                    if lt is None or _isStandaloneGraphIO(buf):
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

        if _censusOn:
            _promotedNames = {n for n, _ in promoted}
            _zeroQ = _zeroQB = 0
            _unadmitted = _unadmittedB = 0
            for _n, _b, _sz, _acc in candidates:
                if _n in _promotedNames:
                    continue
                if _acc * _sz <= 0:
                    _zeroQ += 1
                    _zeroQB += _sz
                else:
                    _unadmitted += 1
                    _unadmittedB += _sz
            _censusN['no measured traffic (Q=0)'] = _zeroQ
            _censusB['no measured traffic (Q=0)'] = _zeroQB
            _censusN['eligible but did not fit'] = _unadmitted
            _censusB['eligible but did not fit'] = _unadmittedB
            # Coverage: how much of the graph's L3 footprint the harvest can even see,
            # and how much is left at L3 after promotion. Without this the reported
            # percentage is a share of the measured subset, not of the real traffic.
            _allL3 = _allL3B = 0
            _leftL3 = _leftL3B = 0
            _seen = set(self.fetchBytes or {})
            _coveredB = _uncoveredB = 0
            for _nm, _bf in list(ctxt.globalObjects.items()) + list(ctxt.localObjects.items()):
                if getattr(_bf, '_memoryLevel', None) != 'L3':
                    continue
                try:
                    _sz = self._bufferSize(_bf)
                except Exception:
                    continue
                _leftL3 += 1
                _leftL3B += _sz
                if _nm in _seen:
                    _coveredB += _sz
                else:
                    _uncoveredB += _sz
            print(f"  [PromoteTensorsToL2] CENSUS still-at-L3 n={_leftL3} bytes={_leftL3B:,} "
                  f"(harvest-covered {_coveredB:,}, harvest-blind {_uncoveredB:,})")
            print(f"  [PromoteTensorsToL2] CENSUS candidates={len(candidates)} "
                  f"promoted={len(promoted)}")
            for _r in sorted(_censusN, key = lambda k: -_censusB[k]):
                if _censusN[_r]:
                    print(f"  [PromoteTensorsToL2] CENSUS   {_r:<28} "
                          f"n={_censusN[_r]:<5} bytes={_censusB[_r]:,}")

        print(f"  [PromoteTensorsToL2] promoted {len(promoted)} tensors, "
              f"{l2_used} / {self.l2Budget} bytes (already={already_l2}, "
              f"new_const={const_used}, var_peak={final_var_peak}, "
              f"strategy={self.strategy!r})")

        # Off-chip traffic the promotion removes. Only meaningful against measured
        # per-tensor traffic: derived from consuming-node counts it would report the
        # schedule-granularity figure, which understates every tensor an operator
        # re-reads once per tile.
        if self.fetchBytes:
            savedBytes = sum(self.fetchBytes.get(name, 0) for name, _ in promoted)
            totalBytes = sum(self.fetchBytes.values())
            share = 100 * savedBytes / totalBytes if totalBytes else 0.0
            print(f"  [PromoteTensorsToL2] off-chip traffic removed: {savedBytes} B "
                  f"of {totalBytes} B ({share:.1f}%), measured")

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
