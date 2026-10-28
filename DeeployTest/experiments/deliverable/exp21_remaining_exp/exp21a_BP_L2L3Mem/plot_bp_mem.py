#!/usr/bin/env python3
# exp21a: live bytes per schedule step (steps.csv from analyze_bp_mem.py) -> bp_mem_timeline.png -- QW
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
r = list(csv.DictReader(open(os.path.join(D, "steps.csv"))))
s = [int(x["step"]) for x in r]
g = lambda k: [int(x[k]) / 1000 for x in r]
inp, pool, arena, l3 = g("l2_promoted_inputs"), g("l2_promoted_pool_live"), g("l2_tile_arena_live"), g("l3_arena_live")
C = {"inp": "#2a78d6", "pool": "#eb6834", "arena": "#1baf7a", "l3": "#6250d6", "ink": "#52514e"}
fig, ax = plt.subplots(2, 1, figsize = (11, 7), sharex = True, facecolor = "#fcfcfb")
for a in ax:
    a.set_facecolor("#fcfcfb"); a.grid(axis = "y", color = "#e4e3df", lw = 0.8); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.tick_params(colors = C["ink"])
ax[0].stackplot(s, inp, pool, arena, colors = [C["inp"], C["pool"], C["arena"]], edgecolor = "#fcfcfb", lw = 0.5,
                labels = ["promoted graph inputs (99.8 kB, standalone mallocs)",
                          "PROMOTED_POOL_L2 live (whole tensors, never in L3)",
                          "MEMORYARENA_L2 live (L3->L2 tile staging, double-buffered)"])
for y, t in ((1024.552, "allocated L2 = 1,024,552 B  (99,752 + pool 687,488 + tile arena 237,312)"),
             (787.240, "live-bytes peak = 787,240 B (step 10, tile arena empty)"), (1300, "--l2 budget 1,300,000 B")):
    ax[0].axhline(y, color = C["ink"], lw = 1, ls = "--"); ax[0].text(0.5, y + 12, t, color = C["ink"], fontsize = 8)
ax[0].set_ylabel("L2 [kB]"); ax[0].set_ylim(0, 1400); ax[0].legend(loc = "upper right", bbox_to_anchor = (1, 0.84), fontsize = 8, frameon = False)
ax[0].set_title("BP single step on GAP9 (exp9 memsim/bp): bytes live per schedule step", loc = "left", fontsize = 11)
ax[1].fill_between(s, l3, color = C["l3"], step = None, lw = 0, label = "MEMORYARENA_L3 live")
ax[1].axhline(1005.648, color = C["ink"], lw = 1, ls = "--")
ax[1].text(0.5, 1005.648 + 12, "allocated L3 arena = 1,005,648 B (+832 B BN running stats)", color = C["ink"], fontsize = 8)
ax[1].axhline(63.372, color = C["ink"], lw = 0.8, ls = ":")
ax[1].text(0.5, 63.372 + 12, "persistent: grad accumulators 61,956 + L3 params 1,412 + flag 4", color = C["ink"], fontsize = 8)
ax[1].annotate("block-0 conv out + BN out kept\nfor backward (2 x 314,048 B)", xy = (50, 691.5), xytext = (55, 420),
               fontsize = 8, color = C["ink"], arrowprops = dict(arrowstyle = "->", color = C["ink"]))
ax[1].annotate("block-0 BN backward:\nX, dX, dY (3 x 314,048 B)", xy = (97, 1005), xytext = (70, 1150), fontsize = 8,
               color = C["ink"], arrowprops = dict(arrowstyle = "->", color = C["ink"]))
ax[1].set_ylabel("L3 [kB]"); ax[1].set_ylim(0, 1400); ax[1].set_xlabel("schedule step (forward 0-41, loss 41-44, backward 44-104)")
ax[1].legend(loc = "upper left", fontsize = 8, frameon = False)
fig.tight_layout(); fig.savefig(os.path.join(D, "bp_mem_timeline.png"), dpi = 150)
