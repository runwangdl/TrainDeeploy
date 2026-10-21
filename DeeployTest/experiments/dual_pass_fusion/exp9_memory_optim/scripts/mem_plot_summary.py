#!/usr/bin/env python3
# exp9 add-on: summarise a Deeploy memory_alloc.html per level: constants (lightblue, whole run) vs variables
# (intermediate tensors with lifetime): peak of live bytes = union of address intervals per step. -- QW
import json, re, sys
def peak(ts):
    if not ts: return 0, None
    steps = int(max(max(t["x"]) for t in ts) + 1); best = (0, None)
    for s in range(steps):
        iv = sorted((min(t["y"]), max(t["y"])) for t in ts if min(t["x"]) <= s <= max(t["x"]))
        tot, cur = 0, None
        for a, b in iv:
            if cur is None or a > cur[1]:
                tot += (cur[1] - cur[0]) if cur else 0; cur = [a, b]
            else: cur[1] = max(cur[1], b)
        tot += (cur[1] - cur[0]) if cur else 0
        best = max(best, (tot, s), key = lambda x: x[0])
    return best
h = open(sys.argv[1]).read()
for lvl in ("L1", "L2", "L3"):
    i = h.find(f'id="plot-{lvl}"')
    if i < 0: continue
    j = h.find("var fig = ", i); k = h.find(";\n", j); fig = json.loads(h[j + 10:k])
    sub = re.sub(r"<[^>]+>|\\u003c[^>]*?\\u003e", " ", fig["layout"].get("title", {}).get("text", ""))
    ts = [t for t in fig["data"] if t.get("x") and t.get("fill") == "toself"]
    const = [t for t in ts if t.get("fillcolor") == "lightblue"]
    var = [t for t in ts if t.get("fillcolor") != "lightblue" and (t.get("line") or {}).get("color") != "red"]
    cbytes = sum(max(t["y"]) - min(t["y"]) for t in const)
    vp, vs = peak(var)
    print(f"{lvl}: const {len(const)} = {cbytes:,} B | var {len(var)} peak live {vp:,} B (step {vs}) | const+var peak {peak(const+var)[0]:,} B")
