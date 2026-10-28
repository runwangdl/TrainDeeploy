# SPDX-License-Identifier: MIT  -- QW
# 2026-08-04  ZO smoke test: generate an isolated single-node PerturbRademacher test
# (network.onnx + inputs.npz + outputs.npz) whose reference is the device-faithful
# packed-Rademacher RNG (bit-exact with TargetLibraries/PULPOpen/src/RandomNoise.c,
# ApplyRademacherPerturbation, for the UNTILED case). Validates the ported op end-to-end.
import os, sys, math
import numpy as np
import onnx
from onnx import helper, TensorProto

NUM_CORES = 8  # matches -DNUM_CORES=8 (Siracusa 8-PE cluster)

def _scramble(seed):
    return np.uint32(np.uint32(seed) * np.uint32(1664525) + np.uint32(1013904223))

def _xorshift32(state):
    s = np.uint32(state)
    s ^= np.uint32(s << np.uint32(13))
    s ^= np.uint32(s >> np.uint32(17))
    s ^= np.uint32(s << np.uint32(5))
    return np.uint32(s)

def perturb_rademacher(data, global_seed, node_id, eps, sign=1):
    """Bit-exact device reference (untiled, tile_seed_offset=0)."""
    flat = data.flatten().astype(np.float32)
    size = flat.size
    log2core = int(math.log2(NUM_CORES))
    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)
        local_size = chunk_stop - chunk_start
        rng_state = _scramble(global_seed + NUM_CORES * node_id + core_id)
        n_full = local_size // 32
        leftover = local_size % 32
        i = chunk_start
        for _ in range(n_full):
            rng_state = _xorshift32(rng_state)
            bits = int(rng_state)
            for _ in range(32):
                rad = np.float32(1.0) if (bits & 1) else np.float32(-1.0)
                flat[i] += np.float32(sign) * rad * np.float32(eps)
                bits >>= 1
                i += 1
        if leftover > 0:
            rng_state = _xorshift32(rng_state)
            bits = int(rng_state)
            for _ in range(leftover):
                rad = np.float32(1.0) if (bits & 1) else np.float32(-1.0)
                flat[i] += np.float32(sign) * rad * np.float32(eps)
                bits >>= 1
                i += 1
    return flat.reshape(data.shape)

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    shape = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["128", "48"])]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    idx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(1234)
    x = rng.randn(*shape).astype(np.float32)
    y = perturb_rademacher(x, seed, idx, eps, sign=1)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node(
        "PerturbRademacher", ["input"], ["output"],
        name="perturb_rademacher_node",
        seed=seed, eps=eps, idx=idx,
    )
    graph = helper.make_graph([node], "perturb_rademacher_graph", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, os.path.join(out_dir, "network.onnx"))
    np.savez(os.path.join(out_dir, "inputs.npz"), input=x)
    np.savez(os.path.join(out_dir, "outputs.npz"), output=y)
    print(f"wrote {out_dir}: shape={shape} seed={seed} eps={eps} idx={idx} size={x.size}")
    print(f"  x[:4]={x.flatten()[:4]}")
    print(f"  y[:4]={y.flatten()[:4]}  (delta first4 = {(y-x).flatten()[:4]})")

if __name__ == "__main__":
    main()
