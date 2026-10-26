"""exp21e BP diagnostic: host argmax signature of fwd mb computed with the INITIAL (never-updated) weights."""
import sys, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa
X, Y = load_windows("speechnet_train_fullfrozen_b1_fold3_ref")
m = build_model(bp_init_map()); am = []
def hook(mod, inp, out):
    x = inp[0].detach(); k = mod.kernel_size[1]; C, H, W = x.shape[1:]; Wo = W // k
    am.append(torch.argmax(x[0, :, :, :Wo*k].reshape(C, H, Wo, k), -1).numpy().ravel())
for mm in m.modules():
    if isinstance(mm, torch.nn.MaxPool2d): mm.register_forward_hook(hook)
with torch.no_grad():
    for mb in range(int(sys.argv[1]), int(sys.argv[2])):
        am = []; l = torch.nn.CrossEntropyLoss()(m(torch.from_numpy(X[mb % 54])), torch.from_numpy(Y[mb % 54])).item()
        a = np.concatenate(am).astype(np.int64) + 1
        print(f"fwd {mb}: initial-weights sig ({a.sum()},{(a*a).sum()})  loss {l:.6f}")
