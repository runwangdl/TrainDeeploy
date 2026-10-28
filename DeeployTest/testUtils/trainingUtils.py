# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers used by the training / optimizer code-generation entry points
(generateTrainingNetwork.py, testMVPTraining.py, generateOptimizerNetwork.py,
testMVPOptimizer.py).

Four kinds of helpers live here, all strictly training-specific:

1. inputs.npz / outputs.npz readers (``_load_reference_losses``, ``_infer_*``).
2. The singleton ``_mockScheduler`` the Tiler expects for per-node tiling.
3. Training-only argparse builders (``add_training_inference_args``,
   ``add_optimizer_training_dir_arg``).
4. The core hooks invoked by ``testUtils.core.execution``
   (``resolve_optimizer_dir``, ``run_training_codegen``,
   ``add_training_cmake_flags``).

Generic helpers (``--cores`` / ``--l1`` / ``--l2`` / ``--defaultMemLevel`` /
``--memAllocStrategy`` / ``--searchStrategy`` / ``--plotMemAlloc`` /
``--profileTiling`` / ``--shouldFail`` arg definitions and the ``shouldFail``
try/except handshake) are deliberately *not* wrapped into functions here:
they are not training-specific and belong inline in whichever entry point
needs them, consistent with the upstream inference codegen scripts.
"""

import argparse
import json
import os
import shutil  # -- QW  copy fixture npz into prepped ZO dirs
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.Logging import DEFAULT_LOGGER as log

# Graph input name marker identifying gradient accumulation buffers.
_GRAD_ACC = "_grad.accumulation.buffer"


def _load_reference_losses(train_dir: str) -> Optional[list]:
    """Load reference loss values from outputs.npz.

    Returns the list of per-mini-batch loss values if any key in
    outputs.npz contains 'loss', otherwise None (with a warning).
    """
    outputs_path = os.path.join(train_dir, "outputs.npz")
    if not os.path.exists(outputs_path):
        log.warning(f"outputs.npz not found at {outputs_path} — loss comparison skipped")
        return None

    try:
        outputs = np.load(outputs_path)
    except Exception as e:
        log.warning(f"Failed to load outputs.npz: {e} — loss comparison skipped")
        return None

    for key in outputs.files:
        if 'loss' in key.lower():
            vals = [float(v) for v in np.array(outputs[key]).flatten().tolist()]
            log.info(f"Reference losses loaded from outputs.npz['{key}']: {vals}")
            return vals

    log.warning("No 'loss' key found in outputs.npz — loss comparison skipped")
    return None


def _infer_num_data_inputs(inputs_path: str) -> int:
    """Auto-detect number of data inputs from inputs.npz.

    Data inputs are the base arr_* entries that have per-mini-batch
    variants (mb1_arr_*) in the npz — i.e. entries that actually change
    across mini-batches.

    Raises ValueError if no mb1 entries are found (single-mini-batch case)
    where the data/weight boundary cannot be determined automatically.
    """
    inputs = np.load(inputs_path)
    base_keys = sorted(k for k in inputs.files if not k.startswith('mb') and not k.startswith('meta_'))
    count = sum(1 for k in base_keys if f'mb1_{k}' in inputs.files)
    if count == 0:
        raise ValueError("Cannot auto-detect num_data_inputs: inputs.npz has only one mini-batch "
                         "(no mb1_arr_* entries found). Please pass --num-data-inputs explicitly.")
    return count


def _infer_total_mb(inputs_path: str) -> int:
    """Count total mini-batches from inputs.npz.

    New format: inputs.npz contains meta_n_batches (total training mini-batches)
    and meta_data_size (number of unique samples stored; C harness cycles via modulo).

    Legacy format: count 1 + number of unique mb* indices.
    """
    inputs = np.load(inputs_path)
    if "meta_n_batches" in inputs.files:
        return int(inputs["meta_n_batches"].flat[0])
    mb_indices = set()
    for key in inputs.files:
        if key.startswith('mb'):
            try:
                idx = int(key.split('_')[0][2:])
                mb_indices.add(idx)
            except ValueError:
                pass
    return 1 + len(mb_indices)


def _infer_data_size(inputs_path: str) -> int:
    """Return the number of unique input samples stored in inputs.npz.

    New format: reads meta_data_size.
    Legacy format: same as _infer_total_mb (all batches were unique).
    """
    inputs = np.load(inputs_path)
    if "meta_data_size" in inputs.files:
        return int(inputs["meta_data_size"].flat[0])
    return _infer_total_mb(inputs_path)


def _infer_n_accum(inputs_path: str) -> int:
    """Return the gradient accumulation step count stored in inputs.npz.

    New format: reads meta_n_accum written by the exporter.
    Legacy format: defaults to 1 (no gradient accumulation).
    """
    inputs = np.load(inputs_path)
    if "meta_n_accum" in inputs.files:
        return int(inputs["meta_n_accum"].flat[0])
    return 1


def deferPerturbNodes(nodes: List[gs.Node]) -> List[gs.Node]:  # -- QW
    """Re-order a topologically-sorted node list so each ZO perturbation node runs
    immediately before its earliest consumer (ported from Deeploy zo-support). -- QW

    Perturb* nodes produce a non-in-place perturbed COPY of a weight; that copy is
    live from the perturb until its consumer. Because perturb nodes depend only on
    constants, Deeploy's toposort HOISTS all of them to the front, keeping all ~22
    copies alive across the whole forward. The SB-tiler then has almost no L2 left,
    tiles every conv/BN into tiny pieces, and the (cycle-accurate) sim crawls
    (~min/forward instead of ~s). Moving each perturb to just before its consumer
    caps peak liveness at ~1 copy. No-op for BP graphs (no Perturb nodes). -- QW
    """  # -- QW
    isPerturb = lambda n: "Perturb" in n.op  # -- QW
    perturb = [n for n in nodes if isPerturb(n)]  # -- QW
    if not perturb:  # -- QW  BP fast path: nothing to defer
        return nodes  # -- QW
    base = [n for n in nodes if not isPerturb(n)]  # -- QW
    pos = {id(n): i for i, n in enumerate(base)}  # -- QW
    before = {id(n): [] for n in base}  # -- QW
    leftover = []  # -- QW
    for p in perturb:  # -- QW
        consumers = [c for c in p.outputs[0].outputs if id(c) in pos]  # -- QW
        if not consumers:  # -- QW
            leftover.append(p)  # -- QW
            continue  # -- QW
        earliest = min(consumers, key = lambda c: pos[id(c)])  # -- QW
        before[id(earliest)].append(p)  # -- QW
    out = list(leftover)  # -- QW
    for n in base:  # -- QW
        out.extend(before[id(n)])  # -- QW
        out.append(n)  # -- QW
    return out  # -- QW


def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    """Wrap every node in a singleton list for the Tiler pattern interface.

    Applies deferPerturbNodes first so ZO perturb copies don't all live at once
    (BP graphs are unaffected — no Perturb nodes). -- QW
    """
    return [[node] for node in deferPerturbNodes(list(graph.nodes))]  # -- QW


# Suffix marking a node, tensor or constant created as a recompute clone. Also the
# signal that a graph has already been injected, so the second scheduler call
# replays the cached order instead of cloning a second time.
_CLONE_MARKER = "__ck"

# Fraction of a schedule's entries that must name a node in the deployed graph before
# the replay is trusted. Not 100%: lowering legitimately renames a few nodes.
_MIN_SCHEDULE_COVERAGE = 0.9


def _loadRecomputeSchedule(path: str) -> List[Tuple[str, bool]]:
    """Read a recompute schedule: a list of (node name, is_recompute) in run order.

    The file is what a Checkmate solve emits, one entry per execution rather than
    per node, so a tensor that is recomputed appears more than once. The first
    appearance of a name is its original execution; every later one asks for a
    fresh clone.
    """
    with open(path) as handle:
        data = json.load(handle)
    return [(str(name), bool(isRecompute)) for name, isRecompute in data["seq"]]


def _liveOutputs(node: gs.Node) -> int:
    return len([out for out in node.outputs if out is not None])


def _nodesByName(graph: gs.Graph) -> Dict[str, gs.Node]:
    """Map name to node, preferring one that still produces something.

    Lowering can leave a stale same-named node whose outputs have been cleared. A
    plain comprehension may pick that one, and scheduling a zero-output node puts
    an empty layer into layerBinding, which fails with an IndexError far from here.
    """
    byName: Dict[str, gs.Node] = {}
    for node in graph.nodes:
        if node.name not in byName or (_liveOutputs(node) and not _liveOutputs(byName[node.name])):
            byName[node.name] = node
    return byName


def _spliceMissingNodes(schedule: List[gs.Node],
                        graph: gs.Graph,
                        excludedNames: Optional[Set[str]] = None) -> List[gs.Node]:
    """Insert nodes the schedule does not mention after their last producer.

    The scheduler runs twice, once for binding and once for the tiler, and lowering
    between the two can add or rename nodes. Dropping them would leave their outputs
    unproduced. Splicing each one in after its last scheduled predecessor keeps the
    rest of the order byte-identical, which a full re-topologisation would not.

    ``excludedNames`` are nodes deliberately left out, which must not be spliced back.
    ``_pruneDeadRecomputes`` removes a node from the SCHEDULE but cannot remove it from
    ``graph.nodes``, so "in the graph and not in the schedule" describes a pruned node
    just as well as a genuinely missing one. Without this, every node the prune dropped
    is reinserted, cleanup strips the outputs nothing reads, and the empty layer reaches
    layerBinding as ``IndexError: list index out of range`` at TilerExtension:1748 --
    the very failure the prune exists to prevent. The two functions are each correct
    alone and cancel each other out when composed.
    """
    excluded = excludedNames or set()
    placed = {id(node) for node in schedule}
    for node in [n for n in graph.nodes if id(n) not in placed and n.name not in excluded]:
        producedNames = {t.name for t in node.inputs if t is not None and t.name}
        insertAt = 0
        for index, scheduled in enumerate(schedule):
            if any(out is not None and out.name in producedNames for out in scheduled.outputs):
                insertAt = index + 1
        schedule.insert(insertAt, node)
    return schedule


def _pruneDeadRecomputes(schedule: List[gs.Node], graph: gs.Graph) -> Tuple[List[gs.Node], Set[str]]:
    """Drop scheduled nodes nothing reads, to a fixpoint.

    A solve may place a recompute in a stage whose consumers linearise before the
    clone, so nothing is rewired to it. It may equally rewire every consumer of an
    original to a clone, which makes the "recompute" a move and leaves the original
    unread. Either way cleanup strips the output and the empty layer reaches
    layerBinding. Dropping such a node is strictly better: less compute and less
    memory. Removing one can orphan its producer, hence the fixpoint.
    """
    graphOutputs = {out.name for out in graph.outputs if out is not None}
    droppedNames: Set[str] = set()
    while True:
        position = {id(node): index for index, node in enumerate(schedule)}
        dead = []
        for node in schedule:
            if any(out is not None and out.name in graphOutputs for out in node.outputs):
                continue
            isRead = any(
                id(consumer) in position and position[id(consumer)] > position[id(node)] for out in node.outputs
                if out is not None for consumer in out.outputs)
            if not isRead:
                dead.append(node)
        if not dead:
            return schedule, droppedNames
        deadIds = {id(node) for node in dead}
        schedule = [node for node in schedule if id(node) not in deadIds]
        droppedNames.update(node.name for node in dead)


def recomputeScheduler(schedulePath: str, shareConstants: bool = True):
    """Build a scheduler that replays a recompute schedule on the deployed graph.

    Gradient checkpointing trades compute for memory: an activation is dropped after
    its forward use and regenerated before the backward one reads it. The decision of
    what to drop comes from a solve over the training graph; this replays that
    decision on the graph Deeploy actually deploys, which is not the graph the solve
    saw, because lowering has run in between.

    Each recompute becomes a short-lived clone, and consumers scheduled after it read
    the clone rather than the original, so the original's output dies at its last
    forward use instead of staying live across the backward pass.

    ``shareConstants`` lets a clone reference the same constant as the original
    rather than carrying a copy. A copy charges the full constant again for every
    clone, so recomputing a node that reads a weight grows the persistent region
    instead of shrinking the peak: measured on the int8 QLoRA CCT, 117 clones pushed
    persistent from 662 KB to 938 KB. Sharing is sound because a ConstantBuffer is
    global and read-only.

    The returned callable is used twice on the same graph, once for binding and once
    for the tiler. The second call replays the order cached from the first: cloning
    again would produce a second set of clones for the same recomputes.
    """
    sequence = _loadRecomputeSchedule(schedulePath)
    cachedOrder: List[Optional[List[str]]] = [None]
    # Carried across the two calls: the tiler call replays the cached order and splices
    # too, so a node the binding call pruned would be reinserted there instead.
    prunedNames: Set[str] = set()

    def schedule(graph: gs.Graph) -> List[List[gs.Node]]:
        byName = _nodesByName(graph)

        if cachedOrder[0] is not None and any(_CLONE_MARKER in node.name for node in graph.nodes):
            replayed = [byName[name] for name in cachedOrder[0] if name in byName]
            replayed = _spliceMissingNodes(replayed, graph, prunedNames)
            return [[node] for node in replayed if _liveOutputs(node)]

        # A schedule is keyed on node names, and lowering renames nodes. If a solve is
        # replayed against a graph it was not produced from, most names miss, every
        # recompute silently vanishes, and what runs is the default order under a name
        # that claims to be checkpointed. That degradation is invisible in a pass/fail
        # CI result, so refuse it rather than report a green run of the wrong thing.
        matched = sum(1 for name, _ in sequence if name in byName)
        if matched < len(sequence) * _MIN_SCHEDULE_COVERAGE:
            raise ValueError(f"Recompute schedule matches only {matched} of {len(sequence)} entries against this "
                             f"graph ({matched / max(len(sequence), 1):.0%}, need "
                             f"{_MIN_SCHEDULE_COVERAGE:.0%}). It was almost certainly produced from a different "
                             f"graph: replaying it would drop the recomputes and silently run the default order.")

        current: Dict[str, gs.Tensor] = {}
        ordered: List[gs.Node] = []
        clones: List[gs.Node] = []
        alreadyRun: Set[str] = set()
        instance = 0

        for name, isRecompute in sequence:
            original = byName.get(name)
            if original is None:
                continue
            rewiredInputs = [current[t.name] if (t is not None and t.name in current) else t for t in original.inputs]

            if not isRecompute and name not in alreadyRun:
                original.inputs = rewiredInputs
                ordered.append(original)
                for out in original.outputs:
                    if out.name:
                        current[out.name] = out
                alreadyRun.add(name)
                continue

            instance += 1
            cloneInputs = []
            for tensor in rewiredInputs:
                if isinstance(tensor, gs.Constant) and not shareConstants:
                    cloneInputs.append(
                        gs.Constant(name = f"{tensor.name}{_CLONE_MARKER}{instance}", values = tensor.values))
                else:
                    cloneInputs.append(tensor)
            cloneOutputs = [
                gs.Variable(name = f"{out.name}{_CLONE_MARKER}{instance}", dtype = out.dtype, shape = out.shape)
                if out.name else out for out in original.outputs
            ]
            clone = gs.Node(op = original.op,
                            name = f"{original.name}{_CLONE_MARKER}{instance}",
                            attrs = dict(original.attrs),
                            inputs = cloneInputs,
                            outputs = cloneOutputs)
            ordered.append(clone)
            clones.append(clone)
            for out, cloneOut in zip(original.outputs, cloneOutputs):
                if out.name:
                    current[out.name] = cloneOut

        ordered, droppedNames = _pruneDeadRecomputes(ordered, graph)
        prunedNames.update(droppedNames)
        if droppedNames:
            log.info(f"[Recompute] dropped {len(droppedNames)} recomputes nothing reads")

        liveClones = {id(node) for node in ordered}
        graph.nodes.extend([node for node in clones if id(node) in liveClones])

        # Nodes the schedule never named still have to run, and they cannot simply go
        # at the end: a constant duplicated per consumer during frontend folding
        # produces a tensor its consumer reads, so appending it after that consumer
        # leaves the input unresolved and context lookup fails. Splice each one in
        # after whatever produces its inputs.
        ordered = _spliceMissingNodes(ordered, graph, prunedNames)

        cachedOrder[0] = [node.name for node in ordered]
        log.info(f"[Recompute] {len(sequence)} scheduled executions, "
                 f"{len([n for n in ordered if _CLONE_MARKER in n.name])} recompute clones")
        return [[node] for node in ordered if _liveOutputs(node)]

    return schedule


# Element size assumed by _tensorSize.  Deliberately uniform: this is the weight
# of a tie-break heuristic, not an allocation, and a training graph is fp32
# almost everywhere.  Reading tensor.dtype instead is not the free improvement it
# looks like -- CCT carries tensors whose ONNX dtype is UNDEFINED, which surfaces
# as 0 and makes np.dtype() raise, and any dtype-aware weighting would produce a
# different order from the one the numbers in this docstring were measured on.
_ASSUMED_ELEMENT_BYTES = 4


def _tensorSize(tensor: Optional[gs.Tensor]) -> int:
    """Approximate byte size of a tensor.

    Dimensions that are not positive integers (symbolic or unknown) count as 1
    rather than collapsing the tensor to 0, so a partially-annotated tensor is
    still ranked above an empty one instead of being ignored outright.
    """
    if tensor is None or getattr(tensor, "shape", None) is None:
        return 0
    count = 1
    for dim in tensor.shape:
        count *= dim if isinstance(dim, int) and dim > 0 else 1
    return count * _ASSUMED_ELEMENT_BYTES


def _memoryMinimisingScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    """Topological order chosen to hold fewer tensors live at once.

    ``_mockScheduler`` emits the graph in the order the exporter happened to
    produce, which for a training graph is roughly "the whole forward pass, then
    the whole backward pass".  That order keeps every saved activation live
    across the entire backward pass even where the consuming gradient node runs
    early, and the arena has to be sized for the resulting peak.

    This is the standard greedy list schedule: among the nodes whose inputs are
    all available, run the one that frees the most memory, where

        net_free(n) = bytes of inputs whose LAST remaining use is n
                    - bytes of outputs n allocates

    Ties break on the node's original index, so the result is deterministic and
    stays as close to the exported order as the objective allows.  The output is
    always a valid topological order -- a node only becomes ready once every
    producer it depends on has run -- so this changes *when* tensors are alive,
    never *what* is computed.  ``InPlaceAccumulatorV2`` writes through to its
    accumulator, so it is charged no allocation.

    Measured (GAP9, ``--defaultMemLevel L3``, L3 peak, address-deduplicated):

        ResNet8       1420 KB -> 1316 KB  (-7.3%)
        MobileNetV1   3903 KB -> 3667 KB  (-6.0%)

    with no rematerialisation and no change to the arithmetic.
    """
    nodes = list(graph.nodes)
    originalIndex = {id(node): i for i, node in enumerate(nodes)}

    # Only tensors produced inside the graph are schedulable state; graph inputs
    # and constants are live throughout regardless of the order we pick.
    producedBy = {output.name: node for node in nodes for output in node.outputs if output is not None and output.name}

    remainingUses = {}
    for node in nodes:
        for tensor in node.inputs:
            if tensor is not None and tensor.name in producedBy:
                remainingUses[tensor.name] = remainingUses.get(tensor.name, 0) + 1

    # Count DISTINCT producers, not input slots: a node consuming the same
    # tensor twice must not need it delivered twice before it becomes ready.
    unmetDeps = {
        id(node): len({tensor.name for tensor in node.inputs if tensor is not None and tensor.name in producedBy})
        for node in nodes
    }

    def netFree(node: gs.Node) -> int:
        freed = sum(
            _tensorSize(tensor)
            for tensor in node.inputs
            if tensor is not None and remainingUses.get(tensor.name, 0) == 1)
        if node.op == "InPlaceAccumulatorV2":
            allocated = 0
        else:
            allocated = sum(_tensorSize(output) for output in node.outputs if output is not None)
        return freed - allocated

    ready = [node for node in nodes if unmetDeps[id(node)] == 0]
    readySet = {id(node) for node in ready}
    schedule = []

    while ready:
        ready.sort(key = lambda node: (-netFree(node), originalIndex[id(node)]))
        node = ready.pop(0)
        readySet.discard(id(node))
        schedule.append(node)

        for tensor in node.inputs:
            if tensor is not None and tensor.name in remainingUses:
                remainingUses[tensor.name] -= 1

        for output in node.outputs:
            if output is None:
                continue
            for consumer in output.outputs:
                if id(consumer) not in unmetDeps or id(consumer) in readySet:
                    continue
                unmetDeps[id(consumer)] -= 1
                if unmetDeps[id(consumer)] == 0:
                    ready.append(consumer)
                    readySet.add(id(consumer))

    # A cycle, or a consumer reachable from a producer but absent from
    # graph.nodes, would silently drop nodes from the generated code.
    assert len(schedule) == len(nodes), \
        f"scheduled {len(schedule)} of {len(nodes)} nodes; graph is not a DAG over graph.nodes"
    return [[node] for node in schedule]


# ---------------------------------------------------------------------------
# argparse builders
#
# The four training / optimizer codegen entry points all define the same
# arguments in their __main__ blocks.  These helpers add the shared groups
# to an existing parser so each entry point only has to compose the groups
# it actually needs.
# ---------------------------------------------------------------------------


def add_training_inference_args(parser: argparse.ArgumentParser) -> None:
    """Arguments consumed by both training codegen entry points."""
    parser.add_argument(
        "--num-data-inputs",
        type = int,
        dest = "num_data_inputs",
        default = None,
        help = "Number of DATA inputs that change per mini-batch. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-steps",
        type = int,
        dest = "n_steps",
        default = None,
        help = "N_TRAIN_STEPS: number of gradient-accumulation update steps. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-accum",
        type = int,
        dest = "n_accum",
        default = None,
        help = "N_ACCUM_STEPS: number of mini-batches per update step. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--learning-rate",
        type = float,
        dest = "learning_rate",
        default = 0.001,
        help = "SGD learning rate emitted as TRAINING_LEARNING_RATE in testinputs.h. Default: 0.001.",
    )
    parser.add_argument(
        "--tolerance",
        type = float,
        dest = "tolerance_abs",
        default = 1e-3,
        help = "Absolute loss tolerance emitted as TRAINING_TOLERANCE_ABS in testoutputs.h. Default: 1e-3.",
    )


def add_optimizer_training_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-dir",
        type = str,
        default = None,
        help = "Directory containing the training network.onnx.  When provided, "
        "weight and grad-acc buffers are shared with TrainingNetwork instead "
        "of being allocated independently.",
    )


def resolve_optimizer_dir(test_dir: str, optimizer_dir: Optional[str]) -> str:
    """Return the optimizer ONNX directory for a training test.

    If ``optimizer_dir`` is explicitly set, it is returned as-is.  Otherwise
    fall back to ``<test_dir>/../<model>_optimizer``, where ``<model>`` is
    derived by replacing the ``_train`` suffix of the test directory's base
    name with ``_optimizer`` (e.g. ``simplemlp_train`` → ``simplemlp_optimizer``,
    ``sleepconvit_train`` → ``sleepconvit_optimizer``).
    """
    if optimizer_dir:
        return optimizer_dir
    test_path = Path(test_dir)
    optimizer_name = test_path.name.replace("_train", "_optimizer")
    return str(test_path.parent / optimizer_name)


def add_training_cmake_flags(cmd: List[str], training: bool, n_train_steps: Optional[int], n_accum_steps: Optional[int],
                             training_num_data_inputs: Optional[int]) -> None:
    """Append -DTRAINING=ON/OFF plus any known -DN_TRAIN_STEPS / -DN_ACCUM_STEPS /
    -DTRAINING_NUM_DATA_INPUTS defines to ``cmd``.  In-place."""
    cmd.append(f"-DTRAINING={'ON' if training else 'OFF'}")
    if not training:
        return
    if n_train_steps is not None:
        cmd.append(f"-DN_TRAIN_STEPS={n_train_steps}")
    if n_accum_steps is not None:
        cmd.append(f"-DN_ACCUM_STEPS={n_accum_steps}")
    if training_num_data_inputs is not None:
        cmd.append(f"-DTRAINING_NUM_DATA_INPUTS={training_num_data_inputs}")


def resolve_zo_update_dir(test_dir: str) -> str:  # -- QW
    """Return the zo_update ONNX directory for a ZO (MeZO) test. -- QW

    Derives the update-graph directory from the zo_train test directory by
    replacing the ``_zo_train`` suffix of the base name with ``_zo_update``
    (e.g. ``speechnet_zo_train`` -> ``speechnet_zo_update``). -- QW
    """  # -- QW
    test_path = Path(test_dir)  # -- QW
    update_name = test_path.name.replace("_zo_train", "_zo_update")  # -- QW
    return str(test_path.parent / update_name)  # -- QW


def add_mezo_cmake_flags(cmd: List[str], config) -> None:  # -- QW
    """Append -DMEZO_TRAINING=ON plus ZO hyperparameter -D's to ``cmd``. In-place. -- QW

    ZO_EPS / ZO_LR / ZO_Q / ZO_SEED come from ``config`` (falling back to the C
    harness defaults when unset). The float defines carry an ``f`` suffix so the
    C preprocessor produces float32 literals. -- QW
    """  # -- QW
    cmd.append("-DMEZO_TRAINING=ON")  # -- QW
    # BP TRAINING must be OFF so the MEZO branch is selected; execution.py's
    # add_training_cmake_flags already emitted -DTRAINING=OFF for a mezo config
    # (config.training stays False), so nothing to undo here. -- QW
    eps = config.zo_eps if config.zo_eps is not None else 0.01  # -- QW
    lr = config.zo_lr if config.zo_lr is not None else 3e-6  # -- QW
    cmd.append(f"-DZO_EPS={eps:.10g}f")  # -- QW
    cmd.append(f"-DZO_LR={lr:.10g}f")  # -- QW
    cmd.append(f"-DZO_Q={int(config.zo_q)}")  # -- QW
    cmd.append(f"-DZO_SEED={int(config.zo_seed)}")  # -- QW
    # SpeechNet ZO trains BN gamma/beta but uses FROZEN pretrained running stats
    # (the reference sim runs model.eval()). BatchNormInternal must therefore use
    # frozen stats, not batch stats — enable it by default for the ZO path so the
    # device matches the reference. -- QW
    cmd.append("-DBN_FROZEN_STATS=ON")  # -- QW
    # N_TRAIN_STEPS / N_ACCUM_STEPS / TRAINING_NUM_DATA_INPUTS are consumed by
    # the MEZO branch too; emit them from the (possibly auto-detected) config. -- QW
    if config.n_train_steps is not None:  # -- QW
        cmd.append(f"-DN_TRAIN_STEPS={config.n_train_steps}")  # -- QW
    if config.n_accum_steps is not None:  # -- QW
        cmd.append(f"-DN_ACCUM_STEPS={config.n_accum_steps}")  # -- QW
    if config.training_num_data_inputs is not None:  # -- QW
        cmd.append(f"-DTRAINING_NUM_DATA_INPUTS={config.training_num_data_inputs}")  # -- QW


def run_zo_codegen(config, script_dir: Path) -> None:  # -- QW
    """Drive the two-stage MeZO (ZO) codegen pipeline for one test. -- QW

    Near-clone of ``run_training_codegen`` (BP), but:
      * Stage 1 runs ``testMVPTraining.py --zo`` on the zo_train dir so the
        outputs header emits loss_plus / loss_minus references (not BP loss). -- QW
      * Stage 2 runs ``testMVPOptimizer.py`` on the zo_update dir (resolved via
        ``resolve_zo_update_dir``) with ``--training-dir`` pointing at zo_train,
        so its 22 PerturbRademacher inputs share the training weight buffers
        (in-place update). -- QW

    The BP ``run_training_codegen`` is left untouched. -- QW
    """  # -- QW
    if config.tiling:  # -- QW
        training_script = script_dir / "testMVPTraining.py"  # -- QW
        optimizer_script = script_dir / "testMVPOptimizer.py"  # -- QW
        opt_passthrough = ("--cores", "--l1", "--l2", "--defaultMemLevel", "--memAllocStrategy", "--searchStrategy",  # -- QW
                           "--plotMemAlloc", "--profileTiling", "--profileNodes", "--promoteToL2",  # -- QW
                           "--promoteToL2Strategy", "--promoteToL2IncludeActivations", "--promoteToL2MaxBufferBytes",  # -- QW
                           "--promoteToL2MinBufferBytes", "--promoteToL2Headroom")  # -- QW
        stage = "Tiled MeZO"  # -- QW
    else:  # -- QW
        # Non-tiled ZO codegen scripts are not provided; the ZO smoke bring-up
        # is tiled-only (Siracusa). Fail loudly rather than silently generate BP. -- QW
        raise RuntimeError("run_zo_codegen currently supports the tiled path only "  # -- QW
                           "(use the tiled Siracusa MeZO runner).")  # -- QW

    # --- Onnx4Deeploy emits the ZO train graph in the reference design (trainable weights as
    # INITIALIZERS / Perturb constant bases, BatchNormInternal, canonical 2-output SCE). The device
    # in-place update needs those weights in writable, name-shareable buffers, so we lower them to
    # graph INPUTS here on a COPY (raw fixture + exported artifact untouched); the prepped graph then
    # flows through the EXISTING training + optimizer codegen exactly like a BP graph, and
    # build_shared_buffer_maps can alias zo_update's outputs onto the shared weight inputs. -- QW
    # QW: Onnx4Deeploy's ZO export emits zo_train with the trainable weights already as graph INPUTS
    #     (byte-structurally like a BP training graph), so we consume the fixture DIRECTLY — no deploy-time
    #     promotion. The "make weights inputs" step lives in the exporter (zo_transform), keeping the flow
    #     clean and identical to BP. (build_shared_buffer_maps then aliases zo_update's outputs onto the
    #     zo_train weight inputs for the in-place cross-step update.) -- QW
    train_dir = config.test_dir  # -- QW

    # --- Stage 1: ZO training network (perturb -> fwd -> SoftmaxCE loss). --- -- QW
    cmd = [  # -- QW
        sys.executable,  # -- QW
        str(training_script),  # -- QW
        "-d",  # -- QW
        config.gen_dir,  # -- QW
        "-t",  # -- QW
        train_dir,  # -- QW  prepped zo_train (raw fixture untouched)
        "-p",  # -- QW
        config.platform,  # -- QW
        "--zo",  # -- QW  swap the outputs-header emitter to loss_plus/loss_minus
    ]  # -- QW
    if config.n_train_steps is not None:  # -- QW
        cmd.append(f"--n-steps={config.n_train_steps}")  # -- QW
    if config.n_accum_steps is not None:  # -- QW
        cmd.append(f"--n-accum={config.n_accum_steps}")  # -- QW
    if config.training_num_data_inputs is not None:  # -- QW
        cmd.append(f"--num-data-inputs={config.training_num_data_inputs}")  # -- QW
    if config.verbose > 0:  # -- QW
        cmd.append("-" + "v" * config.verbose)  # -- QW
    if config.debug:  # -- QW
        cmd.append("--debug")  # -- QW
    cmd.extend(config.gen_args)  # -- QW

    log.debug(f"[Execution] {stage} network generation command: {' '.join(cmd)}")  # -- QW
    if subprocess.run(cmd, check = False).returncode != 0:  # -- QW
        raise RuntimeError(f"{stage} network generation failed for {config.test_name}")  # -- QW

    # Read back auto-detected values written by the training generation script. -- QW
    meta_path = Path(config.gen_dir) / "training_meta.json"  # -- QW
    if meta_path.exists():  # -- QW
        with open(meta_path) as f:  # -- QW
            meta = json.load(f)  # -- QW
        config.n_train_steps = meta["n_train_steps"]  # -- QW
        config.n_accum_steps = meta["n_accum_steps"]  # -- QW
        config.training_num_data_inputs = meta["training_num_data_inputs"]  # -- QW
        log.info(f"[Execution] MeZO meta: {meta}")  # -- QW

    # --- Stage 2: ZO update network (22 in-place PerturbRademacher). --- -- QW
    upd_dir = resolve_zo_update_dir(config.test_dir)  # -- QW
    if config.optimizer_dir:  # -- QW  explicit override wins
        upd_dir = config.optimizer_dir  # -- QW
    if not Path(upd_dir).exists():  # -- QW
        log.warning(f"ZO update directory not found: {upd_dir} — skipping update codegen")  # -- QW
        return  # -- QW
    if not optimizer_script.exists():  # -- QW
        log.warning(f"{optimizer_script.name} not found — skipping update codegen")  # -- QW
        return  # -- QW

    upd_prepped = upd_dir  # -- QW  raw native zo_update dir (no prep needed)

    upd_cmd = [  # -- QW
        sys.executable,  # -- QW
        str(optimizer_script),  # -- QW
        "-d",  # -- QW
        config.gen_dir,  # -- QW
        "-t",  # -- QW
        upd_prepped,  # -- QW
        "-p",  # -- QW
        config.platform,  # -- QW
        f"--training-dir={train_dir}",  # -- QW  match prepped train graph names
    ]  # -- QW
    upd_cmd.extend(arg for arg in config.gen_args if any(arg.startswith(p) for p in opt_passthrough))  # -- QW
    if getattr(config, "optimizer_l1", None):  # -- RW  separate L1 budget for the update graph
        upd_cmd = [a for a in upd_cmd if not a.startswith("--l1")] + [f"--l1={config.optimizer_l1}"]  # -- RW
    if not any(arg.startswith("--defaultMemLevel") for arg in upd_cmd):  # -- QW
        upd_cmd.append("--defaultMemLevel=L2")  # -- QW
    if config.verbose > 0:  # -- QW
        upd_cmd.append("-" + "v" * config.verbose)  # -- QW

    log.debug(f"[Execution] {stage} update network generation command: {' '.join(upd_cmd)}")  # -- QW
    if subprocess.run(upd_cmd, check = False).returncode != 0:  # -- QW
        raise RuntimeError(f"{stage} update network generation failed for {config.test_name}")  # -- QW


def run_training_codegen(config, script_dir: Path) -> None:
    """Drive the two-stage training codegen pipeline for one test.

    Runs the training network codegen script (generateTrainingNetwork.py or
    testMVPTraining.py) followed by the matching optimizer codegen script
    (generateOptimizerNetwork.py or testMVPOptimizer.py), and writes back
    any auto-detected training parameters from ``training_meta.json`` into
    ``config``.

    The single entry point keeps ``testUtils.core.execution.generate_network``
    oblivious to training internals — it only has to call this and return.

    Parameters
    ----------
    config : DeeployTestConfig
        The test configuration (must have ``training=True``).  Training
        fields (``n_train_steps``, ``n_accum_steps``,
        ``training_num_data_inputs``) may be updated in-place from the
        training_meta.json written by the codegen script.
    script_dir : Path
        ``DeeployTest/`` — the directory that hosts the four codegen scripts.
    """
    if config.tiling:
        training_script = script_dir / "testMVPTraining.py"
        optimizer_script = script_dir / "testMVPOptimizer.py"
        opt_passthrough = ("--cores", "--l1", "--l2", "--defaultMemLevel", "--memAllocStrategy", "--searchStrategy",
                           "--plotMemAlloc", "--profileTiling", "--profileNodes", "--promoteToL2",
                           "--promoteToL2Strategy", "--promoteToL2IncludeActivations", "--promoteToL2MaxBufferBytes",
                           "--promoteToL2MinBufferBytes", "--promoteToL2Headroom")
        stage = "Tiled training"
    else:
        training_script = script_dir / "generateTrainingNetwork.py"
        optimizer_script = script_dir / "generateOptimizerNetwork.py"
        opt_passthrough = ("--cores", "--l1", "--l2", "--defaultMemLevel")
        stage = "Training"

    # --- Step 1: Training network (forward + backward + accumulation) ---
    cmd = [
        sys.executable,
        str(training_script),
        "-d",
        config.gen_dir,
        "-t",
        config.test_dir,
        "-p",
        config.platform,
    ]
    if config.n_train_steps is not None:
        cmd.append(f"--n-steps={config.n_train_steps}")
    if config.n_accum_steps is not None:
        cmd.append(f"--n-accum={config.n_accum_steps}")
    if config.training_num_data_inputs is not None:
        cmd.append(f"--num-data-inputs={config.training_num_data_inputs}")
    if config.verbose > 0:
        cmd.append("-" + "v" * config.verbose)
    if config.debug:
        cmd.append("--debug")
    cmd.extend(config.gen_args)

    log.debug(f"[Execution] {stage} network generation command: {' '.join(cmd)}")
    if subprocess.run(cmd, check = False).returncode != 0:
        raise RuntimeError(f"{stage} network generation failed for {config.test_name}")

    # Read back auto-detected values written by the training generation script.
    meta_path = Path(config.gen_dir) / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        config.n_train_steps = meta["n_train_steps"]
        config.n_accum_steps = meta["n_accum_steps"]
        config.training_num_data_inputs = meta["training_num_data_inputs"]
        log.info(f"[Execution] Training meta: {meta}")

    # --- Step 2: Optimizer network (SGD) ---
    opt_dir = resolve_optimizer_dir(config.test_dir, config.optimizer_dir)
    if not Path(opt_dir).exists():
        log.warning(f"Optimizer directory not found: {opt_dir} — skipping optimizer codegen")
        return
    if not optimizer_script.exists():
        log.warning(f"{optimizer_script.name} not found — skipping optimizer codegen")
        return

    opt_cmd = [
        sys.executable,
        str(optimizer_script),
        "-d",
        config.gen_dir,
        "-t",
        opt_dir,
        "-p",
        config.platform,
        f"--training-dir={config.test_dir}",
    ]
    opt_cmd.extend(arg for arg in config.gen_args if any(arg.startswith(p) for p in opt_passthrough))
    if getattr(config, "optimizer_l1", None):  # RW  separate L1 budget for the optimizer graph
        opt_cmd = [a for a in opt_cmd if not a.startswith("--l1")] + [f"--l1={config.optimizer_l1}"]
    if not any(arg.startswith("--defaultMemLevel") for arg in opt_cmd):
        opt_cmd.append("--defaultMemLevel=L2")
    if config.verbose > 0:
        opt_cmd.append("-" + "v" * config.verbose)

    log.debug(f"[Execution] {stage} optimizer network generation command: {' '.join(opt_cmd)}")
    # The residency pass wants Q(t) for the TRAINING graph. Both codegen steps
    # inherit DEEPLOY_FETCH_HARVEST and write the same path, and the optimizer runs
    # second -- so leaving it set here silently replaced the training graph's traffic
    # with the optimizer graph's, which is nothing but SGD updates over weights and
    # gradients and contains no activation at all. Harvest the optimizer graph only
    # when asked for it explicitly.
    opt_env = os.environ.copy()
    opt_env.pop("DEEPLOY_FETCH_HARVEST", None)
    if subprocess.run(opt_cmd, check = False, env = opt_env).returncode != 0:
        raise RuntimeError(f"{stage} optimizer network generation failed for {config.test_name}")
