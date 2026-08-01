# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Independent replay of a materialised schedule. Shares no code with the solver.

The solver's M(t) and the schedule it produces come from the same logic, so an error
there is self-consistent and invisible to any check written inside it. This walks the
emitted sequence with plain reference counting and records the peak live bytes.

Semantics are re-derived from the injector's contract rather than imported: a node reads
the CURRENT version of each input, producing an output makes that version current for
every later reader, and a buffer dies when no later execution reads it.

  replay ~= device      -> the solver's M(t) is wrong
  replay ~= M(t)        -> the model is self-consistent and the device diverges
  replay is neither     -> the materialisation does not implement what was solved
"""
import json, sys
import onnx, onnx_graphsurgeon as gs

DEP, SEQ = sys.argv[1], sys.argv[2]
g = gs.import_onnx(onnx.load(DEP))
byname = {n.name: n for n in g.nodes}
seq = json.load(open(SEQ))['seq']

def sz(t):
    if t is None or t.shape is None: return 0
    p = 1
    for d in t.shape: p *= d if isinstance(d, int) and d > 0 else 1
    return p * 4

ALIAS_FREE = {'InPlaceAccumulatorV2'}

# Pass 1: replay the rewiring to learn which buffer version each execution reads and
# writes, so a clone and the original it replaces are distinct buffers.
cur, execs = {}, []
for k, (name, is_rc) in enumerate(seq):
    nd = byname.get(name)
    if nd is None: continue
    ins = [cur.get(t.name) for t in nd.inputs if t is not None and t.name]
    outs = []
    for o in nd.outputs:
        if o is None or not o.name: continue
        vid = (o.name, k)                       # a fresh buffer per execution
        cur[o.name] = vid
        outs.append((vid, 0 if nd.op in ALIAS_FREE else sz(o)))
    execs.append((k, name, [i for i in ins if i], outs))

# Pass 2: how many later executions read each buffer version
# Every produced version must be registered, not only the consumed ones: a version that
# nothing reads never entered the table, so the "refcount hit zero" release never fired
# and it stayed live to the end. Worth a constant 2.6KB on ResNet8 -- 48 BatchNorm
# saved_mean / saved_inv_std outputs -- and it was exactly the offset the replay showed
# against the device before it was fixed.
remaining = {}
for k, name, ins, outs in execs:
    for vid, b in outs: remaining.setdefault(vid, 0)
    for v in ins: remaining[v] = remaining.get(v, 0) + 1

live, cursz, peak, peak_k = {}, 0, 0, None
for k, name, ins, outs in execs:
    for vid, b in outs:                          # allocate this execution's outputs
        if vid not in live:
            live[vid] = b; cursz += b
    if cursz > peak: peak, peak_k = cursz, k
    for vid, b in outs:                      # an output nobody reads dies immediately
        if remaining.get(vid, 0) == 0 and vid in live:
            cursz -= live.pop(vid)
    for v in ins:                                # release inputs with no later reader
        remaining[v] -= 1
        if remaining[v] == 0 and v in live:
            cursz -= live.pop(v)

print(f'replay peak = {peak/1024:.1f}KB at exec {peak_k} of {len(execs)}')
print(f'  live buffers at peak: {sum(1 for _ in live)} (end state), total execs {len(execs)}')
