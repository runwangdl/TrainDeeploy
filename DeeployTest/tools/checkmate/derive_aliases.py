# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Derive the backend's REAL aliasing decisions and emit them for the ILP.

Why this exists. `opt_solve3.py` charges `sizes[i] = sum(bytes of node i's outputs)`
and zeroes that only for a hand-written operator whitelist, `ALIAS_FREE =
{'InPlaceAccumulatorV2'}`. But Deeploy aliases far more than that: on CCT, 50 of the 92
Transpose/Reshape nodes get the SAME address range as their input, i.e. their output
costs nothing. Charging them full size is wrong in two compounding ways.

  1. M(t) is inflated. On CCT the whitelist gives M(t) = 1122 KB against a measured
     activation peak of 834 KB (+34.5%); using the real alias set gives 739 KB (-11.4%).
     The error drops by 67% and, more tellingly, changes sign -- what is left is the
     backend materialising bytes the model cannot see, which is a different problem.

  2. The solver spends recomputes on tensors that are already free -- PHANTOM SAVINGS.
     It drops an aliased Reshape, books the full saving, and the device saves nothing
     because that address was its input's all along; meanwhile the clone's output is a
     genuinely new buffer and the per-clone auxiliary state is genuinely allocated. The
     model goes down and the device goes UP. Measured on two CCT schedules at the same
     budget: the one with 10 aliased Reshape clones out of 28 books 36% of its saving on
     tensors that do not exist, while the schedule that actually realised well has zero
     aliased clones out of 18.

So aliased nodes must be BOTH zero-sized and banned from recompute.

The alias set is read out of a completed deployment's own allocation plot: a node is
aliased when its first real input and its first real output occupy the same address
range. That makes this a DIAGNOSTIC, not a predictor -- you must deploy once to learn
the aliases, then the solver is honest for every subsequent solve on that graph. A
predictive version would derive the same set from Deeploy's aliasing rules statically;
this reads the ground truth instead, which is what you want while establishing whether
aliasing is the explanation at all.

usage: derive_aliases.py <deploy_dir> [out.json]
       <deploy_dir> is a /tmp/pma_<tag> directory: it must contain
       deeployStates/{backend_post_parsing.onnx,memory_alloc.html}
"""
import json
import re
import sys

import onnx
import onnx_graphsurgeon as gs

DEPLOY = sys.argv[1].rstrip('/')
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{DEPLOY}/aliases.json'

g = gs.import_onnx(onnx.load(f'{DEPLOY}/deeployStates/backend_post_parsing.onnx'))

# The plot carries one trace per allocation. Take the FIRST fig only: it is L3, the
# level these peaks are measured at. 'Memory Size' is the arena ceiling marker, not an
# allocation, and would otherwise look like a block spanning the whole address space.
body = open(f'{DEPLOY}/deeployStates/memory_alloc.html').read().split('var fig = ')[1]
addr = {}
for chunk in body.split('{"fill":"toself"')[1:]:
    nm = re.search(r'"name":"([^"]*)"', chunk)
    xm = re.search(r'"x":\[([^\]]*)\]', chunk)
    ym = re.search(r'"y":\[([^\]]*)\]', chunk)
    if not (nm and xm and ym) or 'Memory Size' in nm.group(1):
        continue
    x = [float(v) for v in xm.group(1).split(',')]
    y = [float(v) for v in ym.group(1).split(',')]
    addr.setdefault(nm.group(1), (min(x), max(x), min(y), max(y)))


def rangeOf(tensorName):
    # The plot labels a block with the buffer name, which carries prefixes and suffixes
    # around the tensor name, so this is a containment test rather than a lookup.
    for k, v in addr.items():
        if tensorName in k:
            return v
    return None


def firstReal(tensors):
    return next((t for t in tensors if t is not None and t.name), None)


def outputBytes(node):
    total = 0
    for o in node.outputs:
        if o is None or o.shape is None:
            continue
        count = 1
        for d in o.shape:
            count *= d if isinstance(d, int) and d > 0 else 1
        total += count * 4
    return total


aliased, byOp, bytesSaved = [], {}, 0
for node in g.nodes:
    i, o = firstReal(node.inputs), firstReal(node.outputs)
    if not (i and o):
        continue
    ri, ro = rangeOf(i.name), rangeOf(o.name)
    # Matching ADDRESSES are not enough. The allocator reuses an address as soon as its
    # previous occupant dies, so two unrelated buffers routinely share a range at
    # different times; requiring only the range treats that as aliasing. A real alias is
    # live SIMULTANEOUSLY with what it aliases, so the time intervals must overlap too.
    # On CCT this distinguishes 84 genuine aliases from 2 Relu nodes whose input runs
    # 0->2 and output 2->4 -- adjacent, never co-resident -- worth 384KB that the earlier
    # version wrongly zeroed.
    if ri is not None and ri[2:] == ro[2:] and ri[0] < ro[1] and ro[0] < ri[1]:
        aliased.append(node.name)
        byOp[node.op] = byOp.get(node.op, 0) + 1
        bytesSaved += outputBytes(node)

total = sum(outputBytes(n) for n in g.nodes)
json.dump(
    {
        'deploy_dir': DEPLOY,
        'aliased': aliased,
        'by_op': byOp,
        'aliased_output_bytes': bytesSaved,
        'total_output_bytes': total
    },
    open(OUT, 'w'),
    indent = 1)
print(f'{len(aliased)}/{len(g.nodes)} nodes aliased  {byOp}')
print(f'their outputs: {bytesSaved/1024:.0f}KB of {total/1024:.0f}KB '
      f'({100*bytesSaved/total:.1f}%) that M(t) currently charges and should not')
print(f'wrote {OUT}')
