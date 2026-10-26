"""exp21e BP: how many MaxPool argmax flips (and how much loss change) does a relative weight perturbation of
size sigma cause on one forward? Calibrates the device AMSIG mismatch."""
import sys, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa
X, Y = load_windows("speechnet_train_fullfrozen_b1_fold3_ref")
init = bp_init_map(); am = []
def run(im, mb):
    m = build_model(im); global am; am = []
    def hook(mod, inp, out):
        x = inp[0].detach(); k = mod.kernel_size[1]; C, H, W = x.shape[1:]; Wo = W // k
        am.append(torch.argmax(x[0, :, :, :Wo*k].reshape(C, H, Wo, k), -1).numpy().ravel())
    for mm in m.modules():
        if isinstance(mm, torch.nn.MaxPool2d): mm.register_forward_hook(hook)
    with torch.no_grad():
        l = torch.nn.CrossEntropyLoss()(m(torch.from_numpy(X[mb])), torch.from_numpy(Y[mb])).item()
    a = np.concatenate(am) + 1
    return l, a, (a.sum(), (a*a).sum())
rng = np.random.default_rng(0)
for mb in (4, 5, 9):
    l0, a0, s0 = run(init, mb)
    for sig in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        im = {k: (v if "running" in k else (v * (1 + sig * rng.standard_normal(v.shape))).astype(np.float32)) for k, v in init.items()}
        l, a, s = run(im, mb)
        print(f"fwd {mb} sigma {sig:.0e}: |dloss| {abs(l-l0):.1e}  flipped windows {np.sum(a!=a0):4d}  d_sig {int(s[0])-int(s0[0]):+d}")
