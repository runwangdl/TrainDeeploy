"""exp21e ZO -- re-run the exp18 host reference with controlled changes, compare to the device trace.

variants
  orig     : byte-for-byte replica of Onnx4Deeploy speechnet_exporter._zo_pytorch_reference
             (perturbation done IN PLACE on the master weights: +eps, -2eps, +eps "restore")
  oop      : perturbation OUT OF PLACE (w_pert = w + s*eps*z from an untouched master copy) -- what the
             device kernel ApplyRademacherPerturbation does (TargetLibraries/PULPOpen/src/RandomNoise.c:337)
  oopdev   : oop + the device's fp32 coefficient chain (deeploymezotest.c:149-178: acc+= (Lp-Lm) fp32,
             g = acc/(2*eps*n) fp32, coeff = -lr*g fp32)
  f64      : oop in float64 everywhere (model, perturbation, update) -- "exact-arithmetic" trajectory

usage: python3 zo_host_variants.py <variant> [n_steps]
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa
from common import _perturb_rademacher  # noqa

variant = sys.argv[1]; n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2700
n_accum, eps, lr, seed, q = 4, 0.01, 3e-6, 42, 1
dt = torch.float64 if variant == "f64" else torch.float32
X, Y = load_windows("speechnet_zo_train_b1_fold3_ref"); N = len(X)
init = init_map_from(f"{ZO_SRC}/network_infer.onnx")
model = build_model(init, dt)
nid = zo_node_ids()
targets = [(t.replace(".", "_"), p, nid[t.replace(".", "_")]) for t, p in model.named_parameters() if t.replace(".", "_") in nid]
assert len(targets) == 22
crit = torch.nn.CrossEntropyLoss()

def z_for(p, g, node_id):
    z = _perturb_rademacher(np.zeros(tuple(p.shape), np.float32), int(g), int(node_id), 1.0, 1)
    return torch.from_numpy(np.ascontiguousarray(z)).reshape(p.shape).to(dt)

master = [p.detach().clone() for (_, p, _) in targets]
LP, LM = [], []
t0 = time.time()
with torch.no_grad():
    for s in range(n_steps):
        g = seed + s * q
        zs = [z_for(p, g, n) for (_, p, n) in targets]
        if variant == "oopdev":
            acc = np.float32(0.0)
        else:
            lp_sum = lm_sum = 0.0
        for a in range(n_accum):
            mb = s * n_accum + a
            x = torch.from_numpy(X[mb % N]).to(dt); y = torch.from_numpy(Y[mb % N])
            if variant == "orig":
                for (_, p, _), z in zip(targets, zs): p.add_(z, alpha=eps)
                Lp = float(crit(model(x), y).item())
                for (_, p, _), z in zip(targets, zs): p.add_(z, alpha=-2.0 * eps)
                Lm = float(crit(model(x), y).item())
                for (_, p, _), z in zip(targets, zs): p.add_(z, alpha=eps)
            else:
                e = torch.tensor(eps, dtype=dt)
                for (_, p, _), z, w in zip(targets, zs, master): p.copy_(w + z * e)
                Lp = float(crit(model(x), y).item())
                for (_, p, _), z, w in zip(targets, zs, master): p.copy_(w - z * e)
                Lm = float(crit(model(x), y).item())
                for (_, p, _), w in zip(targets, master): p.copy_(w)
            if variant == "oopdev":
                acc = np.float32(acc + np.float32(np.float32(Lp) - np.float32(Lm)))
            else:
                lp_sum += Lp; lm_sum += Lm
            LP.append(Lp); LM.append(Lm)
        if variant == "oopdev":
            gp = np.float32(acc / np.float32(np.float32(2.0) * np.float32(eps) * np.float32(n_accum)))
            coeff = float(np.float32(-np.float32(lr) * gp))
        else:
            coeff = -lr * ((lp_sum - lm_sum) / (2.0 * eps * n_accum)) / q
        if variant == "orig":
            for (_, p, _), z in zip(targets, zs): p.add_(z, alpha=coeff)
        else:
            for (_, p, _), z, w in zip(targets, zs, master):
                w.add_(z, alpha=coeff); p.copy_(w)
        if s % 100 == 0:
            print(f"{variant} step {s}/{n_steps}  {time.time()-t0:.0f}s", flush=True)
W = {o: (w if variant != "orig" else p).detach().double().numpy().ravel() for (o, p, _), w in zip(targets, master)}
np.savez(f"{B}/results/zo_host_{variant}_{n_steps}.npz", Lp=np.array(LP), Lm=np.array(LM), **{"w_" + k: v for k, v in W.items()})
print("done", variant, n_steps, f"{time.time()-t0:.0f}s")
