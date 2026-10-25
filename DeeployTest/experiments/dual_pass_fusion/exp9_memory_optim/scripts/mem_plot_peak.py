#!/usr/bin/env python3
# exp9: the L2 peak of a Deeploy memory_alloc.html -- step, total (constants + live variables) and the tensors alive at
# that step, largest first. Same trace classification as mem_plot_summary.py. -- QW
#   python3 mem_plot_peak.py <memory_alloc.html> [level=L2] [top=8]
import json, sys
h = open(sys.argv[1]).read(); lvl = sys.argv[2] if len(sys.argv) > 2 else "L2"; top = int(sys.argv[3]) if len(sys.argv) > 3 else 8
i = h.find(f'id="plot-{lvl}"'); j = h.find("var fig = ", i); k = h.find(";\n", j); fig = json.loads(h[j + 10:k])
ts = [t for t in fig["data"] if t.get("x") and t.get("fill") == "toself" and (t.get("line") or {}).get("color") != "red"]
size = lambda t: max(t["y"]) - min(t["y"])
alive = lambda t, s: min(t["x"]) <= s <= max(t["x"])
const = [t for t in ts if t.get("fillcolor") == "lightblue"]
var = [t for t in ts if t.get("fillcolor") != "lightblue"]
steps = int(max(max(t["x"]) for t in ts) + 1)
def live_bytes(s):  # union of address intervals of the variables alive at step s
    iv = sorted((min(t["y"]), max(t["y"])) for t in var if alive(t, s)); tot, cur = 0, None
    for a, b in iv:
        if cur is None or a > cur[1]: tot += (cur[1] - cur[0]) if cur else 0; cur = [a, b]
        else: cur[1] = max(cur[1], b)
    return tot + ((cur[1] - cur[0]) if cur else 0)
s_peak = max(range(steps), key = live_bytes); vpeak = live_bytes(s_peak); cb = sum(size(t) for t in const)
print(f"{lvl} peak at step {s_peak}: variables {vpeak:,} B + constants {cb:,} B = {vpeak + cb:,} B")
liv = sorted((t for t in var if alive(t, s_peak)), key = size, reverse = True)
for t in liv[:top]: print(f"   {size(t):>9,} B  {t.get('name')}")
rest = liv[top:]
if rest: print(f"   {sum(size(t) for t in rest):>9,} B  ({len(rest)} more tensors)")
