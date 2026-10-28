#!/usr/bin/env python3
# Phase 1 sim search: Deeploy-faithful SGD (eff-batch-1, plain SGD, full model
# incl. BN-train) fine-tuning of SpeechNet. Selection on a clean batch-1 val
# split; batch-2 is the untouched held-out test (vs its own 78.33% zero-shot).
import numpy as np, torch, torch.nn as nn, itertools, sys
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy

B1 = "/app/TrainDeeploy/DeeployTest/Tests/Models/speechnet_infer_original"      # batch1 180
B2 = "/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"                       # batch2 180
CKPT = "/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt"

def load(d):
    z = np.load(d + "/inputs.npz")
    return z["input"].astype(np.float32), z["label"].astype(np.int64)   # (N,1,14,700),(N,)

def newmodel():
    m = SpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    return m

def bal_acc(model, X, y):
    model.eval(); pred = np.empty(len(y), np.int64)
    with torch.no_grad():
        for i in range(len(y)):
            pred[i] = model(torch.from_numpy(X[i:i+1])).numpy().argmax(1)[0]
    cls = sorted(set(y.tolist()))
    return float(np.mean([(pred[y==c]==c).mean() for c in cls]))

def strat_partition(y, per_class_val=6, seed=0):
    rng = np.random.RandomState(seed); val=[]; pool=[]
    for c in sorted(set(y.tolist())):
        idx = np.where(y==c)[0].copy(); rng.shuffle(idx)
        val += idx[:per_class_val].tolist(); pool += idx[per_class_val:].tolist()
    return np.array(pool), np.array(val)

def strat_sample(y, idx_pool, per_class, seed=1):
    rng = np.random.RandomState(seed); sel=[]
    for c in sorted(set(y.tolist())):
        pc = idx_pool[y[idx_pool]==c].copy(); rng.shuffle(pc); sel += pc[:per_class].tolist()
    return np.array(sel)

def main():
    torch.manual_seed(0); np.random.seed(0)
    Xb1,yb1 = load(B1); Xb2,yb2 = load(B2)
    pool, val = strat_partition(yb1, per_class_val=6, seed=0)         # val=54 (6/cls), pool=126 (14/cls)
    Xval,yval = Xb1[val], yb1[val]
    zs_val = bal_acc(newmodel(), Xval, yval)
    zs_b2  = bal_acc(newmodel(), Xb2, yb2)
    print(f"zero-shot:  batch1-val={100*zs_val:.2f}%   batch2(test)={100*zs_b2:.2f}%")
    print("="*84)

    DATA = [("10%", 2), ("20%", 4)]          # per-class train windows (2/cls=18, 4/cls=36)
    LRS  = [2e-3, 1e-3, 5e-4]
    EPOCH_MARKS = [5,10,20,30,40]; MAXEP = max(EPOCH_MARKS)
    results = []
    for (dname,pc), lr in itertools.product(DATA, LRS):
        tr = strat_sample(yb1, pool, per_class=pc, seed=1)
        Xtr,ytr = Xb1[tr], yb1[tr]
        m = newmodel()
        opt = torch.optim.SGD(m.parameters(), lr=lr)   # plain SGD, no momentum/wd (Deeploy-match)
        lossf = nn.CrossEntropyLoss()
        curve = {}
        order = np.arange(len(tr))
        rng = np.random.RandomState(123)
        for ep in range(1, MAXEP+1):
            m.train(); rng.shuffle(order)
            for i in order:                              # eff-batch 1: one window per update
                opt.zero_grad()
                out = m(torch.from_numpy(Xtr[i:i+1]))
                lossf(out, torch.from_numpy(ytr[i:i+1])).backward()
                opt.step()
            if ep in EPOCH_MARKS:
                curve[ep] = (bal_acc(m, Xval, yval), bal_acc(m, Xb2, yb2))
        # selection: best val across marks (robust: take val argmax)
        best_ep = max(curve, key=lambda e: curve[e][0])
        v,t = curve[best_ep]
        results.append((dname,pc,lr,best_ep,v,t,curve))
        marks = "  ".join(f"e{e}:val{100*curve[e][0]:.0f}/b2{100*curve[e][1]:.0f}" for e in EPOCH_MARKS)
        print(f"data={dname}({pc}/cls={pc*9}w) lr={lr:<6} | {marks} | "
              f"SEL e{best_ep} val={100*v:.2f}% b2={100*t:.2f}% (Δb2={100*(t-zs_b2):+.2f})")

    print("="*84)
    # overall pick: among configs, choose by val-selected, tie-break fewer steps (smaller pc*best_ep)
    best = max(results, key=lambda r: (round(r[4],4), -(r[1]*r[3])))
    dname,pc,lr,best_ep,v,t,_ = best
    steps = pc*9*best_ep
    print(f"WINNER: data={dname} ({pc*9} win) lr={lr} epochs={best_ep}  (~{steps} eff-batch-1 steps)")
    print(f"  batch1-val={100*v:.2f}%   batch2-test={100*t:.2f}%   vs zero-shot b2={100*zs_b2:.2f}%  "
          f"=> Δ={100*(t-zs_b2):+.2f} pp")

if __name__ == "__main__":
    main()
