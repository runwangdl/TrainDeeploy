"""exp21e BP: device weights after 2 updates ([WDUMP s=1]) vs host PyTorch after 2 updates, per tensor."""
import sys, re, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa
log = sys.argv[1]
X, Y = load_windows("speechnet_train_fullfrozen_b1_fold3_ref")
init = bp_init_map(); m = build_model(init); opt = torch.optim.SGD(m.parameters(), lr=3e-4); crit = torch.nn.CrossEntropyLoss()
w0 = {t.replace(".", "_"): p.detach().clone().numpy().ravel() for t, p in m.named_parameters()}
for s in range(2):
    opt.zero_grad()
    for a in range(4):
        mb = s * 4 + a; crit(m(torch.from_numpy(X[mb])), torch.from_numpy(Y[mb])).backward()
    opt.step()
host = {t.replace(".", "_"): p.detach().numpy().ravel() for t, p in m.named_parameters()}
names = list(host)
d = {}
for l in open(log, errors="ignore"):
    mm = re.match(r"\[WDUMP s=1 wi=(\d+) n=(\d+)\]\s*(.*)", l.strip())
    if mm: d.setdefault(int(mm.group(1)), []).extend(int(h, 16) for h in mm.group(3).split())
for i, k in enumerate(names):
    dv = np.array(d[i], np.uint32).view(np.float32)
    upd = np.abs(host[k] - w0[k]).max()
    print(f"{k:20s} max|dev-host| {np.abs(dv-host[k]).max():.2e}  (host update size {upd:.2e})  ratio {np.abs(dv-host[k]).max()/max(upd,1e-30):.2f}")
