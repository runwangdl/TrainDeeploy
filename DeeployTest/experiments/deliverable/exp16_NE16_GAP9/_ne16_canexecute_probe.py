#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp16 probe: which convs of a graph would NE16Engine.canExecute() accept?

Cheap up-front check, same spirit as exp13's _opdiff.py: answer "does NE16 claim
anything at all?" without a build. Reports per-conv WHY it is accepted/declined, so
the blocker is named rather than inferred.

NOTE: this probes the RAW exported ONNX. Engine coloring actually runs interleaved with
the lowering passes, where Conv+RequantShift may already be merged into RequantizedConv
and the layout is NHWC. Treat this as indicative; the authoritative count comes from
instrumenting a real deployment (see _ne16_color_report.py).
"""
import sys

import onnx
import onnx_graphsurgeon as gs

from Deeploy.Targets.NE16.Engine import NE16Engine


def probe(path: str, enable3x3: bool = True, enableStrides: bool = True) -> int:
    graph = gs.import_onnx(onnx.load(path))
    eng = NE16Engine("NE16", enable3x3 = enable3x3, enableStrides = enableStrides)

    convs = [n for n in graph.nodes if n.op in ("Conv", "RequantizedConv")]
    print(f"\n=== {path}")
    print(f"    enable3x3={enable3x3} enableStrides={enableStrides}; {len(convs)} conv node(s)")
    accepted = 0
    for n in convs:
        ks = n.attrs.get("kernel_shape")
        grp = n.attrs.get("group")
        wt = n.inputs[1] if len(n.inputs) > 1 else None
        isconst = isinstance(wt, gs.Constant)
        wsrc = ("Constant" if isconst else
                f"Variable<-{wt.inputs[0].op}" if wt is not None and wt.inputs else "Variable<-graph_input")
        ok = eng.canExecute(n)
        accepted += bool(ok)

        # name the reasons rather than just the verdict
        reasons = []
        if not isconst:
            reasons.append("weight is not gs.Constant")
        if ks not in ([1, 1], [3, 3]):
            reasons.append(f"kernel_shape {ks} is not [1,1] or [3,3]")
        if n.attrs.get("dilations") != [1, 1]:
            reasons.append(f"dilations {n.attrs.get('dilations')} != [1,1]")
        print(f"    {'ACCEPT' if ok else 'decline'}  k={ks} group={grp} weight={wsrc}"
              + ("" if ok else "   <- " + "; ".join(reasons)))
    print(f"    --> NE16 would claim {accepted}/{len(convs)} convolutions")
    return accepted


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "Tests/Models/Training/SpeechNet/speechnet_qzo12_train/network.onnx",
        "Tests/Models/Training/SpeechNet/speechnet_qzo12_update/network.onnx",
    ]
    total = sum(probe(p) for p in paths)
    print(f"\nTOTAL accepted across all graphs: {total}")
