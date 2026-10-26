"""exp21e step 4 -- compare device traces against the host variants (run after *_host_variants.py).

Writes results/compare_{zo,bp}.txt and plots/{zo,bp}_variants.png
"""
import numpy as np, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift"
FIX = "/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
what = sys.argv[1]


def per_epoch(d, per=54):
    n = len(d) // per
    return np.array([np.median(np.abs(d[e*per:(e+1)*per])) for e in range(n)]), \
           np.array([np.max(np.abs(d[e*per:(e+1)*per])) for e in range(n)])


def ulps(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64); b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


if what == "zo":
    out = open(B + "/results/compare_zo.txt", "w")
    def P(*a):
        s = " ".join(map(str, a)); print(s); out.write(s + "\n")
    t = np.load(B + "/results/trace_zo.npz")
    dev = {"+": t["dev_plus"], "-": t["dev_minus"]}
    V = {"orig (exp18 fixture ref)": {"+": t["ref_plus"], "-": t["ref_minus"]}}
    W = {}
    for v in ("oop", "oopdev", "f64"):
        f = f"{B}/results/zo_host_{v}_2700.npz"
        if os.path.exists(f):
            h = np.load(f); V[v] = {"+": np.float32(h["Lp"]), "-": np.float32(h["Lm"])}
            W[v] = {k[2:]: h[k] for k in h.files if k.startswith("w_")}
    wd = np.load(B + "/results/wdump_zo.npz")
    names = [k[4:] for k in wd.files if k.startswith("dev_")]
    W["orig (exp18 fixture ref)"] = {k: wd["ref_" + k] for k in names}
    Wdev = {k: wd["dev_" + k] for k in names}
    P("ZO: device vs host variants (both passes, 21600 losses)")
    P(f"{'reference':28s} {'median|d|':>10s} {'p99':>9s} {'max':>9s} {'>1e-3':>6s} {'ep0 med':>9s} {'ep199 med':>9s} {'ep0 ulp':>8s} | final w: max|dw| median ulp")
    curves = {}
    for nm, r in V.items():
        d = np.concatenate([dev["+"].astype(np.float64) - r["+"], dev["-"].astype(np.float64) - r["-"]])
        u = np.concatenate([ulps(dev["+"][:54], r["+"][:54]), ulps(dev["-"][:54], r["-"][:54])])
        mp, _ = per_epoch(dev["-"].astype(np.float64) - r["-"])
        curves[nm] = mp
        wl = ""
        if nm in W:
            dw = np.concatenate([np.abs(Wdev[k].astype(np.float64) - W[nm][k]) for k in names])
            uu = np.concatenate([ulps(Wdev[k], np.float32(W[nm][k])) for k in names])
            wl = f"{dw.max():.2e} {np.median(uu):.0f}"
        P(f"{nm:28s} {np.median(np.abs(d)):10.2e} {np.percentile(np.abs(d),99):9.2e} {np.abs(d).max():9.2e} {np.sum(np.abs(d)>1e-3):6d} {mp[0]:9.2e} {mp[-1]:9.2e} {np.median(u):8.0f} | {wl}")
    # host-vs-host
    if "f64" in V:
        P("\nhost variant vs f64 'exact' trajectory (L-):")
        for nm in V:
            if nm == "f64": continue
            mp, mx = per_epoch(V[nm]["-"].astype(np.float64) - V["f64"]["-"])
            P(f"  {nm:28s} ep0 med {mp[0]:.2e}  ep199 med {mp[-1]:.2e}  max {mx.max():.2e}")
        mp, mx = per_epoch(dev["-"].astype(np.float64) - V["f64"]["-"])
        P(f"  {'DEVICE':28s} ep0 med {mp[0]:.2e}  ep199 med {mp[-1]:.2e}  max {mx.max():.2e}")
    plt.figure(figsize=(7, 4.2))
    for nm, c in curves.items():
        plt.loglog(np.arange(1, len(c)+1), c, label=f"device - {nm}")
    plt.xlabel("epoch (54 forwards)"); plt.ylabel("median |L- device - ref| per epoch"); plt.legend(fontsize=7)
    plt.title("exp18 ZO: which reference does the device follow?"); plt.tight_layout(); plt.savefig(B + "/plots/zo_variants.png", dpi=110)
    out.close()

if what == "bp":
    out = open(B + "/results/compare_bp.txt", "w")
    def P(*a):
        s = " ".join(map(str, a)); print(s); out.write(s + "\n")
    t = np.load(B + "/results/trace_bp.npz")
    ref = np.float32(t["ref_npz"]); dev = t["dev_printed"]
    H = {v: np.load(f"{B}/results/bp_host_{v}.npz") for v in ("orig", "ulp1", "ulp2", "f64", "ulp1F", "ulp2F") if os.path.exists(f"{B}/results/bp_host_{v}.npz")}
    o = H["orig"]
    P(f"BP host 'orig' replica vs exp17 fixture: bit-equal {np.sum(np.float32(o['loss']).view(np.uint32)==ref.view(np.uint32))}/{len(ref)}  max|d| {np.abs(np.float32(o['loss'])-ref).max():.2e}")
    def onset(d, th): i = np.where(np.abs(d) >= th)[0]; return int(i[0]) if len(i) else -1
    P(f"\n{'pair':26s} {'1e-5 onset':>10s} {'1e-4':>6s} {'1e-3':>6s} {'max':>9s} {'median':>9s} {'>1e-3':>6s} | first argmax flip (fwd, #windows)")
    pairs = [("device - fixture", dev - ref, None),
             ("orig(1thr) - fixture(10thr)", np.float64(o["loss"]) - ref, (np.array([]), None))]
    if "f64" in H:
        pairs.append(("device - f64", dev - np.float64(H["f64"]["loss"]), None))
    for v in ("ulp1", "ulp2", "f64", "ulp1F", "ulp2F"):
        if v in H:
            fl = (np.where((H[v]["argmax"] != o["argmax"]).any(1))[0],)
            nfl = (H[v]["argmax"] != o["argmax"]).sum(1)
            pairs.append((f"{v} - orig", np.float64(H[v]["loss"]) - np.float64(o["loss"]), (fl[0], nfl)))
    for nm, d, fl in pairs:
        s = f"{nm:26s} {onset(d,1e-5):10d} {onset(d,1e-4):6d} {onset(d,1e-3):6d} {np.abs(d).max():9.2e} {np.median(np.abs(d)):9.2e} {np.sum(np.abs(d)>1e-3):6d}"
        if fl is not None and fl[1] is not None:
            f0 = fl[0]
            s += f" | first flip fwd {f0[0] if len(f0) else -1}; fwd with >=1 flip: {len(f0)}; first 8: {list(f0[:8])}"
        P(s)
    # near-tie scan in the reference (orig) trajectory
    ties = o["ties"]   # (2160, 3, 4): exact, <1e-6, <1e-5, min rel gap
    P("\nreference MaxPool near-ties (orig trajectory): forwards with a positive-max window whose top-2 relative gap < 1e-6:")
    nt = np.where(ties[:, :, 1].sum(1) > 0)[0]
    P(f"  count {len(nt)} / 2160; exact positive ties in any forward: {int(ties[:, :, 0].sum())}; first 20: {list(nt[:20])}")
    P(f"  min rel gap over fwd 600..640: {ties[600:640, :, 3].min():.2e} (fwd {600 + ties[600:640, :, 3].min(1).argmin()})")
    # per-forward minimum gap around the device onset
    for lo, hi in ((600, 640), (940, 1012)):
        g = ties[lo:hi, :, 3].min(1)
        idx = np.argsort(g)[:5]
        P(f"  fwd {lo}-{hi}: smallest rel gaps " + ", ".join(f"fwd {lo+i}: {g[i]:.1e}" for i in idx))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.4))
    for nm, d, _ in pairs:
        ax.semilogy(np.arange(len(d)), np.maximum.accumulate(np.abs(d)) + 1e-8, label=f"running max |{nm}|")
    ax.axhline(1e-3, color="k", ls="--", lw=.8); ax.set_xlabel("forward"); ax.legend(fontsize=7)
    ax.set_title("exp17 BP: device-vs-ref compared with host-vs-host twins")
    plt.tight_layout(); plt.savefig(B + "/plots/bp_variants.png", dpi=110)
    out.close()
