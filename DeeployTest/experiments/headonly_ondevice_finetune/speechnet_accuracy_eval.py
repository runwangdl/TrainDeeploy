# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate SpeechNet on-device inference accuracy over 180 samples.

For each sample the script writes a single-sample npz to a temporary test
directory, runs deeployRunner_tiled_siracusa, reconstructs the on-device
output logits from simulation stdout, and accumulates per-class recall to
compute balanced accuracy.

Usage (from DeeployTest/, inside the Deeploy Docker container):
    python speechnet_accuracy_eval.py
    python speechnet_accuracy_eval.py --infer-dir Tests/Models/speechnet_infer \\
                                       --l1 128000 --l2 2000000
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

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--infer-dir",
                    default="Tests/Models/speechnet_infer",
                    help="SpeechNet inference test folder (default: %(default)s)")
parser.add_argument("--l1", type=int, default=128000, help="L1 memory in bytes")
parser.add_argument("--l2", type=int, default=2000000, help="L2 memory in bytes")
args = parser.parse_args()

INFER_DIR = args.infer_dir
TEMP_DIR = "Tests/Models/speechnet_infer_eval_tmp"
RESULTS_PATH = Path("speechnet_accuracy_results.json")

# --------------------------------------------------------------------------- #
# Load dataset
# --------------------------------------------------------------------------- #
inputs_npz = np.load(os.path.join(INFER_DIR, "inputs.npz"))
outputs_npz = np.load(os.path.join(INFER_DIR, "outputs.npz"))

all_inputs = inputs_npz["input"]    # (N, 1, 14, 700) float32
all_outputs = outputs_npz["output"]  # (N, 9)          float32
y_true = inputs_npz["label"].tolist()  # (N,)           int64
N = all_inputs.shape[0]
N_CLASSES = all_outputs.shape[1]

print(f"Dataset: {N} samples, {N_CLASSES} classes")
dist = {c: y_true.count(c) for c in sorted(set(y_true))}
print(f"Label distribution: {dist}\n")

# --------------------------------------------------------------------------- #
# Set up temp test directory (network.onnx is constant across all samples)
# --------------------------------------------------------------------------- #
os.makedirs(TEMP_DIR, exist_ok=True)
shutil.copy(os.path.join(INFER_DIR, "network.onnx"), os.path.join(TEMP_DIR, "network.onnx"))

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def write_sample_npz(i):
    np.savez(os.path.join(TEMP_DIR, "inputs.npz"), input=all_inputs[i:i + 1])
    np.savez(os.path.join(TEMP_DIR, "outputs.npz"), output=all_outputs[i:i + 1])


def run_inference():
    """Run inference runner; return (stdout, returncode)."""
    cmd = [
        sys.executable, "deeployRunner_tiled_siracusa.py",
        "-t", TEMP_DIR,
        "--l1", str(args.l1),
        "--l2", str(args.l2),
        "--cores", "8",
        "-vv",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    lines = []
    with subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = proc.wait()
    return "".join(lines), returncode


def parse_result(stdout, ref_logits):
    """
    Reconstruct on-device logits and return (predicted_class, n_errors).

    The simulator prints a line per mismatch (|diff| > 1e-4):
        Expected:   x.xxxxxx  Actual:   y.yyyyyy  Diff: ... at Index   I in Output 0
    Positions not printed are numerically equal to the reference.
    """
    actual = ref_logits.copy()

    mismatch_re = re.compile(r'Actual:\s*(-?[\d.]+)\s+Diff:.*at Index\s+(\d+)', re.IGNORECASE)
    for m in mismatch_re.finditer(stdout):
        actual[int(m.group(2))] = float(m.group(1))

    err_m = re.search(r'Errors:\s*(\d+)\s+out\s+of', stdout)
    n_errors = int(err_m.group(1)) if err_m else -1

    return int(np.argmax(actual)), n_errors


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
y_pred = []
records = []

for i in range(N):
    print(f"\n{'='*60}")
    print(f"Sample {i+1}/{N}  (true label: {y_true[i]})")
    print(f"{'='*60}")

    write_sample_npz(i)
    stdout, returncode = run_inference()
    if returncode != 0:
        print(f"\nERROR: inference runner failed on sample {i} (exit code {returncode}). Stopping.")
        if records:
            RESULTS_PATH.write_text(json.dumps({"partial": True, "samples_completed": i, "samples": records}, indent=2))
            print(f"Partial results ({i} samples) saved to {RESULTS_PATH}")
        sys.exit(returncode)
    pred, n_errors = parse_result(stdout, all_outputs[i])
    correct = pred == y_true[i]
    y_pred.append(pred)
    records.append({
        "sample": i,
        "true": int(y_true[i]),
        "pred": pred,
        "correct": correct,
        "sim_errors": n_errors
    })
    marker = "CORRECT" if correct else "WRONG"
    print(f"  >> [{i+1:3d}/{N}]  true={y_true[i]}  pred={pred}  {marker}  sim_errors={n_errors}")

# --------------------------------------------------------------------------- #
# Balanced accuracy
# --------------------------------------------------------------------------- #
per_class_recall = []
for c in range(N_CLASSES):
    total_c = sum(1 for t in y_true if t == c)
    correct_c = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
    recall = correct_c / total_c if total_c > 0 else 0.0
    per_class_recall.append(recall)

balanced_acc = sum(per_class_recall) / N_CLASSES
total_correct = sum(r["correct"] for r in records)

print("\n" + "=" * 60)
print("Per-class recall:")
for c, r in enumerate(per_class_recall):
    bar = "#" * int(r * 20)
    print(f"  class {c}: {r:.3f}  |{bar:<20}|")
print(f"\nBalanced accuracy : {balanced_acc:.4f}  ({balanced_acc * 100:.2f}%)")
print(f"Overall accuracy  : {total_correct}/{N} = {total_correct / N:.4f}")
print("=" * 60)

# --------------------------------------------------------------------------- #
# Save JSON results
# --------------------------------------------------------------------------- #
results = {
    "n_samples": N,
    "n_classes": N_CLASSES,
    "balanced_accuracy": balanced_acc,
    "overall_accuracy": total_correct / N,
    "per_class_recall": {str(c): per_class_recall[c] for c in range(N_CLASSES)},
    "samples": records,
}
RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {RESULTS_PATH}")
