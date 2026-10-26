"""exp21e step 1 -- parse the exp17 (BP) and exp18 (ZO) device logs into npz traces.

Run in agitated_hugle:  python3 parse_traces.py
Outputs results/trace_bp.npz, results/trace_zo.npz, results/wdump_{bp,zo}.npz
"""
import re, numpy as np, os, gzip
ROOT = "/app/TrainDeeploy/DeeployTest"
DEL = ROOT + "/experiments/deliverable"
OUT = DEL + "/exp21_remaining_exp/exp21e_numerical_drift/results"
FIX = ROOT + "/Tests/Models/Training/SpeechNet"
os.makedirs(OUT, exist_ok=True)


def wdump(lines, step):
    d = {}
    for l in lines:
        m = re.match(r"\[WDUMP s=%d wi=(\d+) n=(\d+)\]\s*(.*)" % step, l)
        if m:
            wi = int(m.group(1))
            d.setdefault(wi, []).extend(int(h, 16) for h in m.group(3).split())
    return {wi: np.array(v, np.uint32).view(np.float32) for wi, v in d.items()}


# ---------------- BP (exp17) ----------------
L = open(DEL + "/exp17_BP_round1_correct_weights/logs/phase3_gvsoc_train_round1.log", errors="ignore").read().splitlines()
comp, ref = {}, {}
for l in L:
    m = re.match(r"\[loss (\d+)\] computed=(\S+)\s+ref=(\S+)", l)
    if m:
        i = int(m.group(1)); comp[i] = float(m.group(2)); ref[i] = float(m.group(3))
n = max(comp) + 1
dev = np.array([comp[i] for i in range(n)]); rprint = np.array([ref[i] for i in range(n)])
o = np.load(FIX + "/speechnet_train_fullfrozen_b1_fold3_ref/outputs.npz")
rnpz = o["loss"].astype(np.float64)
print("BP: n=%d  max|printed ref - npz ref| = %.2e (fixture = the one the run used?)" % (n, np.abs(rprint - rnpz).max()))
wd = wdump(L, 539)
np.savez(OUT + "/trace_bp.npz", dev_printed=dev, ref_printed=rprint, ref_npz=o["loss"])
names = [k for k in o.files if k != "loss" and "running" not in k]
np.savez(OUT + "/wdump_bp.npz", **{"dev_" + names[i]: wd[i] for i in wd if i < len(names)},
         **{"ref_" + k: o[k].ravel() for k in names})
print("BP WDUMP tensors:", len(wd), " ref trainable tensors:", len(names))

# ---------------- ZO (exp18) ----------------
L = open(DEL + "/exp18_ZO_round1_correct_weights/logs/phase3_gvsoc_zo_round1.log", errors="ignore").read().splitlines()
lp = [int(m.group(1), 16) for l in L for m in [re.search(r"lp_bits=0x([0-9a-f]{8})", l)] if m and "ZTRACE" not in l]
lm = [int(m.group(1), 16) for l in L for m in [re.search(r"lm_bits=0x([0-9a-f]{8})", l)] if m and "ZTRACE" not in l]
lp = np.array(lp, np.uint32).view(np.float32); lm = np.array(lm, np.uint32).view(np.float32)
o = np.load(FIX + "/speechnet_zo_train_b1_fold3_ref/outputs.npz")
rp, rm = o["loss_plus"], o["loss_minus"]
pr = {}
for l in L:
    m = re.match(r"\[loss([+-]) (\d+)\] computed=(\S+)\s+ref=(\S+)", l)
    if m:
        pr[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
chk_p = max(abs(pr[("+", i)][1] - rp[i]) for i in range(len(rp)) if ("+", i) in pr)
chk_m = max(abs(pr[("-", i)][1] - rm[i]) for i in range(len(rm)) if ("-", i) in pr)
chk_d = max(abs(pr[("+", i)][0] - lp[i]) for i in range(len(lp)) if ("+", i) in pr)
print("ZO: n+=%d n-=%d  printed-ref vs npz: %.1e %.1e ; printed-dev vs bits: %.1e" % (len(lp), len(lm), chk_p, chk_m, chk_d))
wd = wdump(L, 2699)
names = [k for k in o.files if k not in ("log_prob", "loss_plus", "loss_minus")]
np.savez(OUT + "/trace_zo.npz", dev_plus=lp, dev_minus=lm, ref_plus=rp, ref_minus=rm)
np.savez(OUT + "/wdump_zo.npz", **{"dev_" + names[i]: wd[i] for i in wd if i < len(names)},
         **{"ref_" + k: o[k].ravel() for k in names})
print("ZO WDUMP tensors:", len(wd), " ref trainable tensors:", len(names))
