"""exp21e/fix: device weights after S updates ([WDUMP s=S-1]) vs host PyTorch (exp17 recipe) after S updates, per
tensor; also reports the device-vs-host difference relative to the size of the host update.
usage (agitated_hugle): python3 wdump_compare.py <device log> <S>"""
import sys, re, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa
log, S = sys.argv[1], int(sys.argv[2])
X, Y = load_windows("speechnet_train_fullfrozen_b1_fold3_ref")
init = bp_init_map(); m = build_model(init); opt = torch.optim.SGD(m.parameters(), lr=3e-4); crit = torch.nn.CrossEntropyLoss()
w0 = {t.replace(".", "_"): p.detach().clone().numpy().ravel() for t, p in m.named_parameters()}
for s in range(S):
    opt.zero_grad()
    for a in range(4):
        mb = s * 4 + a; crit(m(torch.from_numpy(X[mb])), torch.from_numpy(Y[mb])).backward()
    opt.step()
host = {t.replace(".", "_"): p.detach().numpy().ravel() for t, p in m.named_parameters()}
d = {}
for l in open(log, errors="ignore"):
    mm = re.match(r"\[WDUMP s=%d wi=(\d+) n=(\d+)\]\s*(.*)" % (S - 1), l.strip())
    if mm: d.setdefault(int(mm.group(1)), []).extend(int(h, 16) for h in mm.group(3).split())
for i, k in enumerate(host):
    dv = np.array(d[i], np.uint32).view(np.float32)
    upd = np.abs(host[k] - w0[k]).max()
    e = np.abs(dv - host[k]); j = int(e.argmax())
    print(f"{k:20s} n={dv.size:5d} max|dev-host| {e.max():.2e} at {j:5d}  host-update {upd:.2e}  ratio {e.max()/max(upd,1e-30):.3f}"
          f"  #(>1e-7) {(e>1e-7).sum()}")
