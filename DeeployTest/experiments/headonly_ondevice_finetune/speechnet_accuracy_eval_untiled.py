# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate SpeechNet on-device inference accuracy over 180 samples using
the untiled Siracusa runner.

For each sample the script writes a single-sample npz to a temporary
test directory, runs deeployRunner_siracusa.py, reads the on-device
output logits from the Logit[i]: lines printed by deeploytest.c, and
accumulates per-class recall to compute balanced accuracy.

Usage (from DeeployTest/, inside the Deeploy Docker container):
    python speechnet_accuracy_eval_untiled.py
    python speechnet_accuracy_eval_untiled.py \\
        --infer-dir Tests/Models/speechnet_infer_revised \\
        --cores 8
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
                    default="Tests/Models/speechnet_infer_revised",
                    help="SpeechNet inference test folder (default: %(default)s)")
parser.add_argument("--cores", type=int, default=8, help="Number of cluster cores (default: 8)")
args = parser.parse_args()

INFER_DIR = args.infer_dir
TEMP_DIR = "Tests/Models/speechnet_infer_eval_tmp"
RESULTS_PATH = Path("speechnet_accuracy_results_untiled.json")

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
    np.savez(os.path.join(TEMP_DIR, "inputs.npz"),
             input=all_inputs[i:i + 1],
             label=np.array([y_true[i]]))
    np.savez(os.path.join(TEMP_DIR, "outputs.npz"),
             output=all_outputs[i:i + 1])


def run_inference():
    """Run untiled inference runner; return (stdout, returncode)."""
    cmd = [
        sys.executable, "deeployRunner_siracusa.py",
        "-t", TEMP_DIR,
        "--cores", str(args.cores),
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


LOGIT_RE = re.compile(r'Logit\[(\d+)\]:\s*(-?[\d.eE+\-]+)')


def parse_result(stdout):
    """
    Parse on-device logits from the Logit[i]: lines printed unconditionally
    by deeploytest.c, return (predicted_class, n_errors).

    Returns (None, -1) if no logit lines are found (inference failed).
    """
    logits = {}
    for m in LOGIT_RE.finditer(stdout):
        logits[int(m.group(1))] = float(m.group(2))

    if not logits:
        return None, -1

    arr = [logits[i] for i in range(len(logits))]

    err_m = re.search(r'Errors:\s*(\d+)\s+out\s+of', stdout)
    n_errors = int(err_m.group(1)) if err_m else -1

    return int(np.argmax(arr)), n_errors


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

    pred, n_errors = parse_result(stdout)

    if pred is None:
        print(f"\nWARNING: could not parse logits for sample {i} "
              f"(exit code {returncode}). Skipping.")
        records.append({
            "sample": i,
            "true": int(y_true[i]),
            "pred": None,
            "correct": False,
            "sim_errors": n_errors,
            "failed": True,
        })
        continue

    correct = pred == y_true[i]
    y_pred.append((i, pred))
    records.append({
        "sample": i,
        "true": int(y_true[i]),
        "pred": pred,
        "correct": correct,
        "sim_errors": n_errors,
        "failed": False,
    })
    marker = "CORRECT" if correct else "WRONG"
    print(f"  >> [{i+1:3d}/{N}]  true={y_true[i]}  pred={pred}  {marker}  sim_errors={n_errors}")

# --------------------------------------------------------------------------- #
# Balanced accuracy
# --------------------------------------------------------------------------- #
successful = [(r["true"], r["pred"]) for r in records if not r.get("failed")]
n_failed = sum(1 for r in records if r.get("failed"))

per_class_recall = []
for c in range(N_CLASSES):
    total_c = sum(1 for t in y_true if t == c)
    correct_c = sum(1 for t, p in successful if t == c and p == c)
    recall = correct_c / total_c if total_c > 0 else 0.0
    per_class_recall.append(recall)

balanced_acc = sum(per_class_recall) / N_CLASSES
total_correct = sum(1 for r in records if not r.get("failed") and r["correct"])

print("\n" + "=" * 60)
if n_failed:
    print(f"WARNING: {n_failed} sample(s) failed (no logits parsed) and were skipped.")
print("Per-class recall:")
for c, r in enumerate(per_class_recall):
    bar = "#" * int(r * 20)
    print(f"  class {c}: {r:.3f}  |{bar:<20}|")
n_evaluated = N - n_failed
print(f"\nBalanced accuracy : {balanced_acc:.4f}  ({balanced_acc * 100:.2f}%)")
print(f"Overall accuracy  : {total_correct}/{n_evaluated} evaluated = {total_correct / n_evaluated:.4f}" if n_evaluated else "No samples evaluated.")
print("=" * 60)

# --------------------------------------------------------------------------- #
# Save JSON results
# --------------------------------------------------------------------------- #
results = {
    "n_samples": N,
    "n_evaluated": n_evaluated,
    "n_failed": n_failed,
    "n_classes": N_CLASSES,
    "balanced_accuracy": balanced_acc,
    "overall_accuracy": total_correct / n_evaluated if n_evaluated else 0.0,
    "per_class_recall": {str(c): per_class_recall[c] for c in range(N_CLASSES)},
    "samples": records,
}
RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {RESULTS_PATH}")
