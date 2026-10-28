# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Compute and print ORT reference weights after 1 optimizer step for b1_ft.
Compare ORT_W[i] lines against DBG_W[i] lines from the on-device simulation
built with DEBUG_WEIGHTS defined.

Usage (from DeeployTest/, in agitated_hugle container):
    python3 speechnet_debug_weights_step1.py
"""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

TRAIN_DIR = Path("Tests/Models/Training/SpeechNet/speechnet_train_b1_ft")
OPT_DIR   = Path("Tests/Models/Training/SpeechNet/speechnet_optimizer")

inp_npz = np.load(TRAIN_DIR / "inputs.npz")

# ── Extract initializer values from the training ONNX ────────────────────────
train_model = onnx.load(str(TRAIN_DIR / "network_train.onnx"))
initializers = {init.name: numpy_helper.to_array(init)
                for init in train_model.graph.initializer}

# ── Build feed by name ───────────────────────────────────────────────────────
train_sess         = ort.InferenceSession(str(TRAIN_DIR / "network_train.onnx"))
train_input_names  = [i.name for i in train_sess.get_inputs()]
train_output_names = [o.name for o in train_sess.get_outputs()]

# arr_* keys cover: input, label, 22 trainable params, lazy_reset_grad
arr_keys = sorted([k for k in inp_npz.files
                   if k.startswith("arr_") and not k.startswith("mb")])

# Names of the 22 trainable weight inputs (positional, excluding running stats)
# Order: data (2) → trainable weights (22) → lazy_reset (1)
# Running stats are initializers in the ONNX, not in arr_*
grad_suffix = "_grad.accumulation.buffer"
weight_input_names = [n for n in train_input_names
                      if not n.endswith(grad_suffix)
                      and "lazy_reset" not in n
                      and "running_mean" not in n
                      and "running_var" not in n]

feed = {}

# 1. Map positional arr_* → weight inputs by position
for name, key in zip(weight_input_names, arr_keys):
    feed[name] = inp_npz[key]

# 2. Running stats: load from network_infer.onnx initializers (pretrained BN stats)
infer_model  = onnx.load(str(TRAIN_DIR / "network_infer.onnx"))
infer_inits  = {i.name: numpy_helper.to_array(i) for i in infer_model.graph.initializer}
for name in train_input_names:
    if ("running_mean" in name or "running_var" in name) and name not in feed:
        if name in infer_inits:
            feed[name] = infer_inits[name]
        else:
            raise RuntimeError(f"No initializer found for {name} in network_infer.onnx")

# 3. Grad accumulation buffers: zero-initialized
for name in train_input_names:
    if name.endswith(grad_suffix) and name not in feed:
        weight_name = name[: -len(grad_suffix)]
        if weight_name in feed:
            feed[name] = np.zeros_like(feed[weight_name])
        else:
            raise RuntimeError(f"Cannot infer shape for grad buffer: {name}")

# 4. lazy_reset_grad = True (first step)
for name in train_input_names:
    if "lazy_reset" in name and name not in feed:
        feed[name] = np.array([True], dtype=bool)

assert set(train_input_names) == set(feed.keys()), \
    f"Missing: {set(train_input_names) - set(feed.keys())}"

# ── Run training network ─────────────────────────────────────────────────────
print("Running training network (step 0, lazy_reset=True)...")
train_outputs = train_sess.run(None, feed)
loss_val = float(train_outputs[0].flat[0])
print(f"  loss = {loss_val:.6f}")

# ── Apply SGD manually (lr=0.001): w_new = w - lr * grad ────────────────────
LR = 0.001

# grads: all training outputs except loss (index 0)
grad_vals = [v for v, n in zip(train_outputs, train_output_names)
             if "loss" not in n.lower()]

# initial weights: arr_0002..arr_0023 (22 trainable params, skip input+label+reset)
weight_vals  = [inp_npz[k] for k in arr_keys[2:-1]]
weight_names = [n for n in train_input_names
                if not n.endswith(grad_suffix)
                and "lazy_reset" not in n
                and "running_mean" not in n
                and "running_var" not in n
                and n not in ("input", "labels")]

print("Applying SGD (lr=0.001)...")
opt_outputs      = [w - LR * g for w, g in zip(weight_vals, grad_vals)]
opt_output_names = [n + "_updated" for n in weight_names]

# ── Print updated weights ─────────────────────────────────────────────────────
print("\nORT reference weights after optimizer step 0 (compare to DBG_W[i]):")
for wi, (name, val) in enumerate(zip(opt_output_names, opt_outputs)):
    flat     = val.flatten()
    n        = len(flat)
    preview  = flat[:16]
    suffix   = f" ...(+{n - 16} more)" if n > 16 else ""
    vals_str = " ".join(f"{v:.8f}" for v in preview)
    print(f"  ORT_W[{wi}] n={n} ({name}): {vals_str}{suffix}")
