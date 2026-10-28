#!/usr/bin/env python3
"""exp2: parse --profileTiling logs -> per-op cycle breakdown + Perfetto Chrome trace.

Handles the fact that with n-accum=4 the forward/backward/accumulator network runs
4x per training step (each op section appears 4x), while the SGD optimizer runs 1x.
Per-op cycles are deterministic across micro-batches, so we take the *first* instance
of each uniquely-named section for the per-op cycle figure, and multiply by its
per-step invocation count to get the per-step total.
"""
import re, sys, json, collections

SECTION_RE = re.compile(r"===== Profiling (\S+) =====")
TILE_RE = re.compile(
    r"\[(?P<node>[^\]]+)\]\[(?P<buf>\w+)\]\[(?P<ops>\d+) ops\]\[Tile (?P<tile>\d+)\]"
    r"\s+(?P<flavor>Pre-Kernel|Kernel\s+|Post-Kernel)\s*:\s*(?P<cycles>\d+)")

N_ACCUM = 4  # forward/backward/accumulator run this many times per training step

def op_type(name):
    # strip the _L2/_L3 suffix and trailing transpose/index noise for grouping
    return name

def classify(name):
    n = name.lower()
    if "sgd" in n or "gradientaccumulator" in n or "inplaceaccumulator" in n or "optimizer" in n:
        return "optimizer"
    # backward: any Grad node, plus transposes/im2col feeding a Grad node
    if "grad" in n or "backward" in n:
        return "backward"
    if "softmaxcrossentropyloss" in n and "grad" not in n:
        return "forward"  # loss forward
    return "forward"

def kernel_family(name):
    """Coarse op family for aggregation."""
    n = name
    for key in ["ConvGradW","ConvGradX","ConvGradB","ConvGrad","Conv",
                "BatchNormalizationGrad","BatchNormInternal","BatchNorm",
                "ReluGrad","Relu","MaxPoolGrad","MaxPool","GlobalAveragePoolGrad",
                "GlobalAveragePool","AveragePool","Gemm",
                "SoftmaxCrossEntropyLossGrad","SoftmaxCrossEntropyLoss",
                "ReduceSum","InPlaceAccumulatorV2","SGDOptimizerV2","sgd","Reshape","transpose"]:
        if key.lower() in n.lower():
            return key
    return "other"

def parse_log(text):
    """Return list of (node_name, {'pre':,'ker':,'post':}) aggregated over tiles,
    one entry per section occurrence."""
    sections = []
    cur = None
    tiles = {}
    for line in text.splitlines():
        m = SECTION_RE.search(line)
        if m:
            if cur is not None:
                sections.append((cur, tiles))
            cur = m.group(1); tiles = {}
            continue
        if cur is None:
            continue
        mt = TILE_RE.search(line)
        if mt:
            t = int(mt.group("tile")); fl = mt.group("flavor").strip(); cyc = int(mt.group("cycles"))
            e = tiles.setdefault(t, {"pre":0,"ker":0,"post":0})
            if fl == "Pre-Kernel": e["pre"] += cyc
            elif fl == "Post-Kernel": e["post"] += cyc
            elif fl == "Kernel": e["ker"] += cyc
    if cur is not None:
        sections.append((cur, tiles))
    return sections

def classify_l3(sections):
    """Deeploy emits a cluster block and an L3 block per node; the L3 block has
    fewer tiles. Mark L3 blocks so we don't double count kernel cycles."""
    node_idx = collections.defaultdict(list)
    for i,(n,_) in enumerate(sections):
        node_idx[n].append(i)
    is_l3 = {}
    for n, idxs in node_idx.items():
        prev = None
        for i in idxs:
            c = len(sections[i][1])
            is_l3[i] = (prev is not None and c < prev)
            prev = c
    return is_l3

def main(logs, out_json):
    # Merge: dedupe by node name, take first occurrence's per-op cycles, count occurrences.
    per_op = {}   # name -> {'pre','ker','post','count','is_l3'}
    for path in logs:
        text = open(path, errors="replace").read()
        secs = parse_log(text)
        is_l3 = classify_l3(secs)
        # count how many times each name occurs in THIS log
        for i,(name, tiles) in enumerate(secs):
            pre = sum(t["pre"] for t in tiles.values())
            ker = sum(t["ker"] for t in tiles.values())
            post = sum(t["post"] for t in tiles.values())
            rec = per_op.setdefault(name, {"pre":0,"ker":0,"post":0,"count":0,"is_l3":is_l3[i]})
            if rec["count"] == 0:
                rec["pre"], rec["ker"], rec["post"], rec["is_l3"] = pre, ker, post, is_l3[i]
            rec["count"] += 1

    # Cluster (non-L3) sections carry compute + L1 DMA. L3 sections carry L3 DMA.
    # Per-step invocation count: forward/backward/accum run N_ACCUM x, optimizer 1x.
    rows = []
    for name, r in per_op.items():
        phase = classify(name)
        invoc = 1 if phase == "optimizer" else N_ACCUM
        rows.append({
            "name": name, "phase": phase, "family": kernel_family(name),
            "is_l3": r["is_l3"], "pre": r["pre"], "ker": r["ker"], "post": r["post"],
            "invoc": invoc,
        })

    # ---- Aggregate cycles per step ----
    def agg(filt):
        compute=dma_l1=dma_l3=0
        for r in rows:
            if not filt(r): continue
            if r["is_l3"]:
                dma_l3 += (r["pre"]+r["post"]) * r["invoc"]
            else:
                compute += r["ker"] * r["invoc"]
                dma_l1 += (r["pre"]+r["post"]) * r["invoc"]
        return compute, dma_l1, dma_l3

    phases = ["forward","backward","optimizer"]
    print("="*78)
    print("PER-STEP CYCLE BREAKDOWN (one training step = %d micro-batches + 1 optimizer)" % N_ACCUM)
    print("="*78)
    grand_c=grand_d1=grand_d3=0
    for ph in phases:
        c,d1,d3 = agg(lambda r,ph=ph: r["phase"]==ph)
        tot=c+d1+d3
        grand_c+=c; grand_d1+=d1; grand_d3+=d3
        print(f"{ph:10s}  compute={c:>10,}  L1dma={d1:>10,}  L3dma={d3:>10,}  TOTAL={tot:>11,}")
    gtot=grand_c+grand_d1+grand_d3
    print("-"*78)
    print(f"{'TOTAL':10s}  compute={grand_c:>10,}  L1dma={grand_d1:>10,}  L3dma={grand_d3:>10,}  TOTAL={gtot:>11,}")
    print(f"\ncompute-fraction = {100*grand_c/gtot:.1f}%   DMA-fraction = {100*(grand_d1+grand_d3)/gtot:.1f}%")

    # ---- Top ops by total per-step cycles (cluster blocks only for op ranking) ----
    op_tot = []
    for r in rows:
        if r["is_l3"]:
            continue
        tot = (r["ker"]+r["pre"]+r["post"]) * r["invoc"]
        op_tot.append((tot, r["ker"]*r["invoc"], (r["pre"]+r["post"])*r["invoc"], r["phase"], r["name"]))
    op_tot.sort(reverse=True)
    print("\n" + "="*78)
    print("TOP 15 OPS BY PER-STEP TOTAL CYCLES (cluster blocks: Kernel + L2<->L1 DMA)")
    print("="*78)
    print(f"{'total':>10} {'kernel':>10} {'dma':>9} {'ph':<9} name")
    for tot,ker,dma,ph,name in op_tot[:15]:
        short = name[:60]
        print(f"{tot:>10,} {ker:>10,} {dma:>9,} {ph:<9} {short}")

    # ---- Family aggregation ----
    fam = collections.defaultdict(lambda: [0,0,0])  # family -> [compute, l1dma, count]
    for r in rows:
        if r["is_l3"]: continue
        fam[r["family"]][0] += r["ker"]*r["invoc"]
        fam[r["family"]][1] += (r["pre"]+r["post"])*r["invoc"]
        fam[r["family"]][2] += 1
    print("\n" + "="*78)
    print("BY OP FAMILY (per step, cluster blocks)")
    print("="*78)
    print(f"{'family':<28} {'compute':>11} {'L1dma':>11} {'total':>11} {'nodes':>6}")
    for f,(c,d,n) in sorted(fam.items(), key=lambda kv:-(kv[1][0]+kv[1][1])):
        print(f"{f:<28} {c:>11,} {d:>11,} {c+d:>11,} {n:>6}")

    # ---- Perfetto Chrome trace ----
    # One track per phase; sequential timeline in cycles (ts in microseconds proxy).
    trace = {"traceEvents": [], "displayTimeUnit": "ns"}
    pid = 1
    tid = {"forward":10,"backward":20,"optimizer":30,"DMA":40}
    for ph,t in tid.items():
        trace["traceEvents"].append({"ph":"M","pid":pid,"tid":t,"name":"thread_name",
                                     "args":{"name":ph}})
    trace["traceEvents"].append({"ph":"M","pid":pid,"name":"process_name","args":{"name":"SpeechNet FT step"}})
    # lay out ops sequentially per phase track
    cursor = {ph:0 for ph in tid}
    for r in sorted(rows, key=lambda r:(r["phase"], r["name"])):
        if r["is_l3"]:
            continue
        dur_k = r["ker"]; dur_dma = r["pre"]+r["post"]
        ph = r["phase"]; t = tid[ph]
        ts = cursor[ph]
        # DMA-in event then kernel then DMA-out on same track for readability
        if r["pre"]:
            trace["traceEvents"].append({"ph":"X","pid":pid,"tid":tid["DMA"],"ts":ts,
                "dur":r["pre"],"name":"DMAin:"+r["family"],"args":{"node":r["name"]}})
        trace["traceEvents"].append({"ph":"X","pid":pid,"tid":t,"ts":ts,"dur":max(dur_k,1),
            "name":r["family"],"args":{"node":r["name"],"kernel":r["ker"],"dma":dur_dma,"invoc":r["invoc"]}})
        cursor[ph] += dur_k + dur_dma
    json.dump(trace, open(out_json,"w"))
    print(f"\nWrote Perfetto trace: {out_json} ({len(trace['traceEvents'])} events)")

if __name__ == "__main__":
    logs = sys.argv[1:-1]; out = sys.argv[-1]
    main(logs, out)
