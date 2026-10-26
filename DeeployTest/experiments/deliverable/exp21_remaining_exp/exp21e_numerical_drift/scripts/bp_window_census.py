"""exp21e BP: census of MaxPool windows on the host reference for given forwards -- how many windows are
'fragile' (max tiny positive, or relative top-2 gap small). Replays orig for the first n_steps."""
import sys, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa
X, Y = load_windows("speechnet_train_fullfrozen_b1_fold3_ref"); N = len(X)
model = build_model(bp_init_map()); params = list(model.parameters()); opt = torch.optim.SGD(params, lr=3e-4)
crit = torch.nn.CrossEntropyLoss(); rec = []
def hook(m, inp, out):
    x = inp[0].detach(); k = m.kernel_size[1]; C, H, W = x.shape[1:]; Wo = W // k
    v = x[0, :, :, :Wo*k].reshape(C, H, Wo, k); s, _ = torch.sort(v, -1, descending=True)
    rec.append((s[..., 0].numpy().ravel(), s[..., 1].numpy().ravel()))
for m in model.modules():
    if isinstance(m, torch.nn.MaxPool2d): m.register_forward_hook(hook)
for mb in range(int(sys.argv[1]) if len(sys.argv) > 1 else 12):
    if mb % 4 == 0: opt.zero_grad()
    rec = []
    loss = crit(model(torch.from_numpy(X[mb % N])), torch.from_numpy(Y[mb % N])); loss.backward()
    if mb % 4 == 3: opt.step()
    t1 = np.concatenate([r[0] for r in rec]); t2 = np.concatenate([r[1] for r in rec])
    pos = t1 > 0
    print(f"fwd {mb:3d}: windows {t1.size}  all-zero {np.sum(t1==0)}  0<max<1e-6 {np.sum(pos & (t1<1e-6))}  0<max<1e-4 {np.sum(pos&(t1<1e-4))}  "
          f"rel-gap<1e-6 {np.sum(pos & ((t1-t2)/np.where(pos,t1,1) < 1e-6))}  <1e-4 {np.sum(pos & ((t1-t2)/np.where(pos,t1,1) < 1e-4))}  min pos max {t1[pos].min():.1e}")
