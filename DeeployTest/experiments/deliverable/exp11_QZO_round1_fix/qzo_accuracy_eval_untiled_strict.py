# SPDX-License-Identifier: MIT
"""exp9 — QZO batch-2 accuracy on the untiled Siracusa device runner (exp5 protocol, quantized).

Mirror of experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py with the two
QZO-specific requirements baked in:
  1. every deeployRunner_siracusa.py call passes `-D BN_FROZEN_STATS=ON` — the quantized
     inference graph carries BatchNormInternal (training op, flag-gated frozen-stats semantics);
     without the flag BN computes single-window batch stats (found in the exp9 smoke, 9/9 errors).
  2. TEST_SIRACUSA is wiped ONCE at start (stale training-runner cmake cache references
     TrainingNetwork.c), then reused across samples for incremental rebuilds.

Consumes the fixture produced by build_qzo_infer_fixture.py (network.onnx + inputs.npz{input,
label} + outputs.npz{output}). Per window: 1-sample temp fixture -> untiled run -> parse
Logit[i] lines -> prediction + bit-exactness errors. Balanced accuracy = mean per-class recall.

Run from DeeployTest/ inside `traindeeploy`:
  python3 experiments/deliverable/exp9_QZO_round1/qzo_accuracy_eval_untiled.py \
      --infer-dir experiments/deliverable/exp9_QZO_round1/qinfer_round1 --cores 8
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--infer-dir", required=True)
parser.add_argument("--cores", type=int, default=8)
parser.add_argument("--start", type=int, default=0, help="resume from window index")
parser.add_argument("--results", default=None)
args = parser.parse_args()

INFER_DIR = args.infer_dir
TEMP_DIR = "Tests/Models/qzo_infer_eval_tmp"
RESULTS_PATH = Path(args.results or (Path(INFER_DIR) / "eval_results.json"))

inputs_npz = np.load(os.path.join(INFER_DIR, "inputs.npz"))
outputs_npz = np.load(os.path.join(INFER_DIR, "outputs.npz"))
all_inputs = inputs_npz["input"]
all_ref = outputs_npz["output"]
y_true = inputs_npz["label"].reshape(-1).tolist()
N = all_inputs.shape[0]
print(f"Dataset: {N} windows, ref accuracy "
      f"{100.0*float((all_ref.argmax(1)==np.array(y_true)).mean()):.2f}%")

os.makedirs(TEMP_DIR, exist_ok=True)
shutil.copy(os.path.join(INFER_DIR, "network.onnx"), os.path.join(TEMP_DIR, "network.onnx"))
if args.start == 0:
    subprocess.run(["rm", "-rf", "TEST_SIRACUSA"], check=False)   # stale training cmake cache
subprocess.run("pgrep -f '[g]vsoc_launcher' | xargs -r kill -9", shell=True, check=False)

LOGIT_RE = re.compile(r"Logit\[(\d+)\]:\s*(-?[\d.eE+\-]+)")
ERR_RE = re.compile(r"Errors:\s*(\d+)\s+out\s+of")


def run_sample(i):
    np.savez(os.path.join(TEMP_DIR, "inputs.npz"),
             input=all_inputs[i:i + 1], label=np.array([y_true[i]]))
    np.savez(os.path.join(TEMP_DIR, "outputs.npz"), output=all_ref[i:i + 1])
    cmd = [sys.executable, "deeployRunner_siracusa.py", "-t", TEMP_DIR,
           "--cores", str(args.cores), "-vv", "-D", "BN_FROZEN_STATS=ON", "DEEPLOY_STRICT_FP32=ON", "DEEPLOY_STRICT_FP32_FILES=BatchNorm.c;Gemm.c;GlobalAveragePool.c;RandomNoise.c"]
    out = subprocess.run(cmd, text=True, capture_output=True,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"}).stdout
    logits = {int(m.group(1)): float(m.group(2)) for m in LOGIT_RE.finditer(out)}
    err_m = ERR_RE.search(out)
    if not logits:
        return None, -1
    arr = [logits[k] for k in range(len(logits))]
    return int(np.argmax(arr)), (int(err_m.group(1)) if err_m else -1)


records = []
if args.start > 0 and RESULTS_PATH.exists():
    records = json.load(open(RESULTS_PATH))["records"][:args.start]

for i in range(args.start, N):
    pred, n_err = run_sample(i)
    ok = pred == y_true[i]
    records.append(dict(i=i, true=int(y_true[i]), pred=(int(pred) if pred is not None else None),
                        bitexact_errors=n_err))
    print(f"[{i+1:3d}/{N}] true={y_true[i]} pred={pred} "
          f"{'OK ' if ok else 'MISS'} bitexact_err={n_err}", flush=True)
    y_p = [r["pred"] for r in records]
    y_t = [r["true"] for r in records]
    rec = {}
    for c in sorted(set(y_t)):
        idx = [k for k, t in enumerate(y_t) if t == c]
        rec[c] = float(np.mean([y_p[k] == c for k in idx]))
    bal = float(np.mean(list(rec.values())))
    json.dump(dict(n_done=len(records), balanced_accuracy=100.0 * bal,
                   overall=100.0 * float(np.mean([p == t for p, t in zip(y_p, y_t)])),
                   n_bitexact_fail=sum(1 for r in records if r["bitexact_errors"] != 0),
                   per_class_recall=rec, records=records),
              open(RESULTS_PATH, "w"), indent=1)

print(f"\nDONE: balanced={100.0*bal:.2f}%  bit-exact fails: "
      f"{sum(1 for r in records if r['bitexact_errors'] != 0)}/{N}  -> {RESULTS_PATH}")
