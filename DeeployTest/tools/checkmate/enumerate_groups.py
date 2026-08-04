"""Exhaustive rematerialisation search over GROUPS of nodes, not individual nodes.

`solve.py` decides per node, which is the right granularity for ResNet8 and the wrong one
for the other two training models we deploy. The n^2 ILP has 204k variables on
MobileNetV1 and returns UNKNOWN after 25 minutes without a feasible schedule; on CCT it
reaches 2772 KB after 46 minutes a point and plateaus. Both networks are built from
repeated structure, so the decision can be made per REPEATED UNIT instead, and the
resulting space is small enough to enumerate exactly:

    network        nodes   solve.py               this script            better
    ResNet8          103   969 KB, 8 points       1217 KB (units)        solve.py, by far
    CCT              244   2772 KB, 46 min/pt     2610 KB, < 1 min       this
    MobileNetV1      281   no feasible solution   2731 KB                this

This is the argument Rockmate makes, and it holds here for the reason Rockmate gives:
repeated structure makes the per-block space small enough to be exact. The corollary is
the useful part -- coarse enumeration is not a fallback for when the ILP is too slow. On
a repetitive graph it finds BETTER schedules, because deciding per node lets the solver
pick combinations that are locally cheap in cycles while leaving large intermediates
straddling the peak.

Scoring uses the same independent replay `replay.py` implements: reference counting over
the materialised sequence, sharing no code with the ILP's memory model. The replay is a
PREDICTION, not a measurement -- CCT's 4-, 5- and 6-unit schedules tie at 672.5 KB in the
replay and the device puts 6 units 33 KB HIGHER, because each surviving clone occupies
memory the replay cannot see. Deploy every candidate you intend to report.

usage:
    enumerate_groups.py <graph.onnx> <partition> [out_prefix]

    [--candidates forward|all]
                  `forward` (default) recomputes only forward nodes, the Rockmate
                  restriction. `all` lifts it. Which one wins is a property of the
                  NETWORK, not of the method -- see the table below.

    <partition>   `blocks`  MobileNetV1: one group per `blocks_blocks_<i>_` block
                  `units`   CCT: attention / MLP / remainder of each transformer block,
                            plus stem+head as one group
                  a regex   any pattern with one capture group; nodes sharing a capture
                            are one group, non-matching nodes form a single `rest` group

Writes <out_prefix>_<k>.json for the best k-group choice at every k, plus a summary
<out_prefix>.json. Feed the JSON to `--recomputeSchedule`.
"""
import collections
import itertools
import json
import re
import sys

import onnx
import onnx_graphsurgeon as gs


# A group is recomputed by re-running its FORWARD nodes immediately before the backward
# node that first needs them. Gradient nodes are never re-run: their inputs are the very
# activations this is trying not to keep, so recomputing one would recompute the forward
# anyway and then discard the saving.
#
# CLASSIFY BY NAME, NOT BY OP. A training graph's backward pass uses the SAME operators as
# its forward: CCT's gradient path is Gemm, MatMul, Transpose and Reshape, exactly like
# its forward path. An op-type whitelist put 113 of CCT's 170 backward nodes -- 38 Gemm,
# 36 Reshape, 28 Transpose -- on the recompute candidate list, so schedules spent their
# budget re-running gradient computations. MobileNetV1 hid this: its backward is ConvGrad
# and ConvGradW, distinct op types, and only 3 nodes were misclassified.
#
# Deeploy suffixes every gradient node with `_backward` or names it `*Grad*`, which is the
# distinction that actually holds.
def isBackward(node):
    return '_backward' in node.name or 'Grad' in node.name


PARTITIONS = {
    'blocks': lambda name: (m.group(1) if (m := re.search(r'blocks_blocks_(\d+)_', name)) else 'rest'),
}


def _cctUnit(name):
    """CCT: two transformer blocks are too coarse (4 candidates), per node too fine.

    Each block splits cleanly into attention, MLP and the residual/norm remainder, and the
    stem plus classifier form a seventh group: 2^7 = 128, still exhaustive. The stem group
    is worth keeping in the enumeration even though it never helps -- recomputing it takes
    the replay from 672.5 to 768.5 KB, because the frozen tokenizer chain must materialise
    ~512 KB of intermediates to avoid retaining a 32 KB endpoint. Seeing that in the sweep
    is more useful than excluding it by hand and wondering later.
    """
    match = re.search(r'blocks_(\d+)_', name)
    if not match:
        return 'stem_head'
    block = match.group(1)
    if 'self_attn' in name:
        return f'blk{block}_attn'
    if re.search(r'linear|mlp|fc', name):
        return f'blk{block}_mlp'
    return f'blk{block}_rest'


PARTITIONS['units'] = _cctUnit


def tensorBytes(tensor):
    if tensor is None or tensor.shape is None:
        return 0
    count = 1
    for dim in tensor.shape:
        count *= dim if isinstance(dim, int) and dim > 0 else 1
    return count * 4


def replayPeak(sequence, byName):
    """Peak live bytes over the materialised sequence, by reference counting.

    Shares no code with the ILP's memory model, which is the entire point: an error in
    that model is self-consistent and invisible from inside it. `InPlaceAccumulatorV2`
    writes through to its accumulator and is charged no allocation.
    """
    current, executions = {}, []
    for step, (name, _) in enumerate(sequence):
        node = byName.get(name)
        if node is None:
            continue
        inputs = [current.get(t.name) for t in node.inputs if t is not None and t.name]
        outputs = []
        for out in node.outputs:
            if out is None or not out.name:
                continue
            version = (out.name, step)
            current[out.name] = version
            outputs.append((version, 0 if node.op == 'InPlaceAccumulatorV2' else tensorBytes(out)))
        executions.append(([v for v in inputs if v], outputs))

    remaining = {}
    for ins, outs in executions:
        for version, _ in outs:
            remaining.setdefault(version, 0)
        for version in ins:
            remaining[version] = remaining.get(version, 0) + 1

    live, size, peak = {}, 0, 0
    for ins, outs in executions:
        for version, nbytes in outs:
            if version not in live:
                live[version] = nbytes
                size += nbytes
        peak = max(peak, size)
        for version, _ in outs:
            if remaining.get(version, 0) == 0 and version in live:
                size -= live.pop(version)
        for version in ins:
            remaining[version] -= 1
            if remaining[version] == 0 and version in live:
                size -= live.pop(version)
    return peak


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    candidates = 'all' if '--candidates=all' in sys.argv or (
        '--candidates' in sys.argv and sys.argv[sys.argv.index('--candidates') + 1] == 'all') else 'forward'
    graphPath, partitionArg = argv[0], argv[1]
    prefix = argv[2] if len(argv) > 2 else 'recompute_groups'

    if partitionArg in PARTITIONS:
        groupOf = PARTITIONS[partitionArg]
    else:
        pattern = re.compile(partitionArg)
        groupOf = lambda name: (m.group(1) if (m := pattern.search(name)) else 'rest')

    graph = gs.import_onnx(onnx.load(graphPath))
    sys.path.insert(0, __file__.rsplit('/', 3)[0])
    from testUtils.trainingUtils import _memoryMinimisingScheduler

    # Enumerate against the order the deployer will actually use. Solving against the
    # exported order and replaying against the scheduled one measures a schedule nobody
    # runs.
    order = [entry[0] for entry in _memoryMinimisingScheduler(graph)]
    position = {node.name: i for i, node in enumerate(order)}
    byName = {node.name: node for node in graph.nodes}

    groups = collections.defaultdict(lambda: {'fwd': [], 'bwd': []})
    for node in order:
        # Two distinct roles, and conflating them silently disables recomputation:
        # `bwd` locates the INSERTION POINT (the first gradient node of the group, before
        # which the recomputes are placed), while `fwd` is the CANDIDATE SET. Putting
        # every node in `fwd` under --candidates=all leaves `bwd` empty, the group gets no
        # insertion point, and nothing is inserted -- the sweep then reports the baseline
        # at every k and looks like "recompute does not help here".
        back = isBackward(node)
        if back:
            groups[groupOf(node.name)]['bwd'].append(node.name)
        if candidates == 'all' or not back:
            groups[groupOf(node.name)]['fwd'].append(node.name)
    keys = sorted(groups)
    print(f'{len(keys)} groups: ' + ', '.join(f'{k}({len(groups[k]["fwd"])}f/{len(groups[k]["bwd"])}b)' for k in keys),
          flush = True)
    if len(keys) > 20:
        sys.exit(f'refusing to enumerate 2^{len(keys)} candidates; use a coarser partition')

    def build(chosen):
        """Place a group's recomputes immediately before the first node that READS them.

        The obvious anchor -- the group's first backward node -- is wrong, and wrong by a
        lot. Which node that is depends on how forward and backward are told apart, and
        the two plausible definitions disagree by up to 177 positions in a 244-step order
        on CCT. Anchoring 177 steps before the consumer keeps every recomputed value alive
        that much longer and gives back the whole saving. The published CCT curve depended
        on an accidentally late anchor, not on a deliberate one.

        The consumer is a property of the TENSORS the group produces, so it is found from
        them: the earliest node outside the group that reads any of the group's outputs
        and runs after the group does.
        """
        insertions = collections.defaultdict(list)
        for key in chosen:
            produced = {
                out.name for name in groups[key]['fwd'] for out in byName[name].outputs if out is not None and out.name
            }
            groupSpan = {position[n] for n in groups[key]['fwd']}
            last = max(groupSpan) if groupSpan else 0
            # Only BACKWARD consumers anchor. A block's output is read by the next
            # block's forward immediately, so anchoring at the first consumer of any kind
            # lands right after the group and saves nothing -- measured: every
            # MobileNetV1 point collapses back to the 1848.4 KB baseline. What
            # rematerialisation avoids is keeping the value alive from the forward use to
            # the backward one, so the backward use is the anchor.
            anchor = None
            for i, node in enumerate(order):
                if i <= last or i in groupSpan or not isBackward(node):
                    continue
                if any(t is not None and t.name in produced for t in node.inputs):
                    anchor = i
                    break
            if anchor is None:  # nothing outside reads it: not worth
                continue  # recomputing, and nothing to anchor to
            insertions[anchor].extend(groups[key]['fwd'])
        sequence = []
        for i, node in enumerate(order):
            for name in insertions.get(i, []):
                sequence.append([name, 1])
            sequence.append([node.name, 0])
        return sequence

    best = []
    for size in range(len(keys) + 1):
        peak, chosen = min(((replayPeak(build(set(c)), byName), c) for c in itertools.combinations(keys, size)),
                           key = lambda x: x[0])
        best.append({'n': size, 'replay_bytes': peak, 'groups': list(chosen)})
        print(f'  recompute {size} groups: {peak/1024:8.1f} KB  {list(chosen)}', flush = True)
        json.dump({'seq': build(set(chosen))}, open(f'{prefix}_{size}.json', 'w'))
    json.dump(best, open(f'{prefix}.json', 'w'), indent = 1)
    print(f'wrote {prefix}_0..{len(keys)}.json and {prefix}.json')
    print('REPLAY IS A PREDICTION. Deploy every point you intend to report: surviving '
          'clones occupy memory this does not model, and the replay has tied points the '
          'device separates by 33 KB.')


if __name__ == '__main__':
    main()
