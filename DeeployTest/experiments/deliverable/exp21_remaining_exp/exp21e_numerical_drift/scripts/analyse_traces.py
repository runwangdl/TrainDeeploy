"""exp21e step 2 -- characterise device-vs-reference loss differences (onset, growth law, pass asymmetry).

Run in agitated_hugle after parse_traces.py.  Writes results/trace_stats.txt and plots/*.png
"""
import numpy as np, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift"
out = open(B + "/results/trace_stats.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

def ulpdiff(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)

def epoch_stats(d, L, per, name):
    ne = len(d) // per
    med = np.array([np.median(np.abs(d[e*per:(e+1)*per])) for e in range(ne)])
    rel = np.array([np.median(np.abs(d[e*per:(e+1)*per]) / np.maximum(np.abs(L[e*per:(e+1)*per]), 1e-3)) for e in range(ne)])
    mx = np.array([np.max(np.abs(d[e*per:(e+1)*per])) for e in range(ne)])
    P(f"--- {name}: per-epoch ({per} fwd) median|d|, median rel |d|/|L|, max|d|")
    for e in list(range(0, min(ne, 6))) + list(range(6, ne, max(1, ne // 12))) + [ne - 1]:
        P(f"  epoch {e:4d}: med {med[e]:.2e}  rel {rel[e]:.2e}  max {mx[e]:.2e}")
    # growth law fit on epochs >= 2 (log-log)
    e = np.arange(1, ne + 1); m = med > 0
    k = np.polyfit(np.log(e[m][2:]), np.log(med[m][2:]), 1)[0]
    k2 = np.polyfit(np.log(e[m][2:]), np.log(rel[m][2:]), 1)[0]
    P(f"  log-log slope median|d| ~ epoch^{k:.2f} ; rel ~ epoch^{k2:.2f}   (0.5 = random walk, 1 = linear)")
    return med, rel, mx

# ======================= ZO =======================
z = np.load(B + "/results/trace_zo.npz")
dp = z["dev_plus"].astype(np.float64) - z["ref_plus"]; dm = z["dev_minus"].astype(np.float64) - z["ref_minus"]
up = ulpdiff(z["dev_plus"], z["ref_plus"]); um = ulpdiff(z["dev_minus"], z["ref_minus"])
P("================ ZO (exp18) raw-bit comparison ================")
for nm, u, d in (("L+", up, dp), ("L-", um, dm)):
    P(f"{nm}: exact {np.mean(u==0)*100:.1f}%  | first 54 fwd: exact {np.mean(u[:54]==0)*100:.0f}%, median ulp {np.median(u[:54]):.0f}, max ulp {u[:54].max()} "
      f"| first 4 fwd ulps {list(u[:4])} | last 54: median ulp {np.median(u[-54:]):.0f}")
    P(f"     median|d| {np.median(np.abs(d)):.2e}  max {np.abs(d).max():.3e} at fwd {np.abs(d).argmax()}  breaches>1e-3: {np.sum(np.abs(d)>1e-3)}")
first_nz_p = np.argmax(up > 0); first_nz_m = np.argmax(um > 0)
P(f"first non-zero ulp diff: L+ fwd {first_nz_p} ({up[first_nz_p]} ulp), L- fwd {first_nz_m} ({um[first_nz_m]} ulp)")
# L- vs L+ asymmetry in the first epoch(s)
P(f"epoch-0 median ulp L+ {np.median(up[:54]):.1f} vs L- {np.median(um[:54]):.1f};  mean |d| L+ {np.mean(np.abs(dp[:54])):.2e} L- {np.mean(np.abs(dm[:54])):.2e}")
mp = epoch_stats(dp, z["ref_plus"], 54, "ZO L+")
mm = epoch_stats(dm, z["ref_minus"], 54, "ZO L-")
# difference signal: (L+ - L-) = what drives g_proj
gd = (z["dev_plus"].astype(np.float64) - z["dev_minus"]) - (z["ref_plus"].astype(np.float64) - z["ref_minus"])
P(f"(L+ - L-) device-minus-ref: median |.| {np.median(np.abs(gd)):.2e}  (vs median |L+-L-| {np.median(np.abs(z['ref_plus']-z['ref_minus'])):.2e})")
# same-window comparison: windows repeat every 54 forwards; follow window w across epochs
P("same-window |d| L- across epochs (window 0..3): " + "; ".join(
    f"w{w}: " + ",".join(f"{abs(dm[w + 54*e]):.1e}" for e in (0, 1, 10, 50, 100, 150, 199)) for w in range(3)))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(dp))
ax[0].semilogy(x, np.abs(dp) + 1e-9, ",", alpha=.4, label="|L+ dev-ref|")
ax[0].semilogy(x, np.abs(dm) + 1e-9, ",", alpha=.4, label="|L- dev-ref|")
ax[0].semilogy(np.arange(len(mp[0])) * 54 + 27, mp[0], "C0-", lw=2, label="L+ per-epoch median")
ax[0].semilogy(np.arange(len(mm[0])) * 54 + 27, mm[0], "C1-", lw=2, label="L- per-epoch median")
ax[0].axhline(1e-3, color="k", ls="--", lw=.8); ax[0].set_xlabel("forward index"); ax[0].set_title("exp18 ZO: device - reference"); ax[0].legend(markerscale=20, fontsize=7)
e = np.arange(1, len(mm[0]) + 1)
ax[1].loglog(e, mp[1], label="L+ median rel"); ax[1].loglog(e, mm[1], label="L- median rel")
ax[1].loglog(e, mm[1][4] * np.sqrt(e / 5), "k:", label="~sqrt(t)"); ax[1].loglog(e, mm[1][4] * (e / 5), "k--", label="~t")
ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7); ax[1].set_title("growth law")
plt.tight_layout(); plt.savefig(B + "/plots/zo_diff_vs_step.png", dpi=110)

# ======================= BP =======================
b = np.load(B + "/results/trace_bp.npz")
d = b["dev_printed"] - b["ref_printed"]   # printed with 6 decimals: resolution 1e-6
L = b["ref_npz"].astype(np.float64)
P("\n================ BP (exp17), 6-decimal printed values ================")
P(f"first 54 fwd: nonzero printed diffs {np.sum(np.abs(d[:54])>1.5e-6)}/54 (>1.5e-6), max {np.abs(d[:54]).max():.1e}, loss range {L[:54].min():.3f}..{L[:54].max():.3f}")
P(f"first fwd with |d|>=2e-6: {np.argmax(np.abs(d)>=1.9e-6)};  >=1e-5: {np.argmax(np.abs(d)>=1e-5)};  >=1e-4: {np.argmax(np.abs(d)>=1e-4)}; >=1e-3: {np.argmax(np.abs(d)>=1e-3)}")
mb = epoch_stats(d, L, 54, "BP")
jumps = np.diff(np.log10(mb[1] + 1e-12))
P(f"largest epoch-to-epoch jump of median rel diff: x{10**jumps.max():.1f} at epoch {jumps.argmax()+1}; median ratio x{10**np.median(jumps):.2f}/epoch")
P("same-window |d| across epochs (window 0..3): " + "; ".join(
    f"w{w}: " + ",".join(f"{abs(d[w + 54*e]):.1e}" for e in (0, 1, 5, 10, 20, 30, 39)) for w in range(3)))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].semilogy(np.abs(d) + 1e-7, ".", ms=2, alpha=.5); ax[0].semilogy(np.arange(len(mb[0])) * 54 + 27, mb[0], "r-", lw=2, label="per-epoch median")
ax[0].axhline(1e-3, color="k", ls="--", lw=.8); ax[0].set_xlabel("forward index (4 per update)"); ax[0].set_title("exp17 BP: |device - reference| (print resolution 1e-6)"); ax[0].legend()
e = np.arange(1, len(mb[1]) + 1)
ax[1].semilogy(e, mb[1], "o-", label="median rel |d|/L"); ax[1].semilogy(e, mb[2], "s-", label="max |d|")
ax[1].set_xlabel("epoch (13.5 updates)"); ax[1].legend(); ax[1].set_title("growth (semilog: straight line = exponential)")
plt.tight_layout(); plt.savefig(B + "/plots/bp_diff_vs_step.png", dpi=110)

# ======================= final weights =======================
for tag in ("bp", "zo"):
    w = np.load(B + f"/results/wdump_{tag}.npz")
    P(f"\n=== final weights device vs ref ({tag}) : per tensor  max|dw|, median ulp, max rel ===")
    for k in [k[4:] for k in w.files if k.startswith("dev_")]:
        a, r = w["dev_" + k], w["ref_" + k]
        u = ulpdiff(a, r)
        P(f"  {k:22s} n={a.size:5d}  max|dw| {np.abs(a-r).max():.2e}  median ulp {np.median(u):8.0f}  max rel {np.max(np.abs(a-r)/np.maximum(np.abs(r),1e-6)):.1e}")
out.close()
