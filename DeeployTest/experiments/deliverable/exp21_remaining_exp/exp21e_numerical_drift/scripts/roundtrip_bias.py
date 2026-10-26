"""exp21e ZO: bias of the reference's in-place perturbation round trip w -> (w+eps*z) -> (.-2eps*z) -> (.+eps*z), fp32,
eps=0.01, z=+-1 (exactly what speechnet_exporter._zo_pytorch_reference does per window), as a function of |w|."""
import numpy as np, torch
rng = np.random.default_rng(0); eps = 0.01
print(f"{'w range':18s} {'mean err':>10s} {'mean err/ulp':>13s} {'mean|err|/ulp':>14s}  x10800 windows")
for lo, hi in [(0.02, 0.115), (0.115, 0.125), (0.125, 0.135), (0.135, 0.24), (0.24, 0.49), (0.49, 0.5), (0.5, 0.51), (0.98, 0.99), (0.99, 1.0), (1.0, 1.01), (1.01, 1.2)]:
    for sgn in (1, -1):
        w = torch.from_numpy((sgn * rng.uniform(lo, hi, 200000)).astype(np.float32)); z = torch.from_numpy(rng.choice([-1., 1.], 200000).astype(np.float32))
        p = w.clone(); p.add_(z, alpha=eps); p.add_(z, alpha=-2 * eps); p.add_(z, alpha=eps)
        e = (p.double() - w.double()).numpy(); u = np.spacing(np.abs(w.numpy())).astype(np.float64)
        print(f"{'+' if sgn>0 else '-'}[{lo},{hi})".ljust(18) + f" {e.mean():+10.2e} {np.mean(e/u):+13.3f} {np.mean(np.abs(e)/u):14.3f}  {e.mean()*10800:+.2e}")
