# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Run SpeechNet training with pretrained weights on tiled Siracusa for 100
optimizer steps and plot the resulting loss curve.

Usage (from DeeployTest/):
    python speechnet_loss_experiment.py
    python speechnet_loss_experiment.py --skipgen   # re-use generated C code
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PREWEIGHTS_DIR = "Tests/Models/Training/SpeechNet/speechnet_train_preweights"
N_STEPS = 100
LOG_PATH = Path("speechnet_100step_run.log")
PLOT_PATH = Path("speechnet_100step_loss.png")

skipgen = "--skipgen" in sys.argv

OPTIMIZER_DIR = "Tests/Models/Training/SpeechNet/speechnet_optimizer"

cmd = [
    sys.executable, "deeployTrainingRunner_tiled_siracusa.py",
    "-t", PREWEIGHTS_DIR,
    "--optimizer-dir", OPTIMIZER_DIR,
    "--l1", "128000",
    "--l2", "2000000",
    "--n-steps", str(N_STEPS),
    "-vv",
]
if skipgen:
    cmd.append("--skipgen")

env = {**os.environ, "PYTHONUNBUFFERED": "1"}

print(f"Running: {' '.join(cmd)}\n")

log_lines = []
with subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, env=env, bufsize=1) as proc:
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)

stdout_text = "".join(log_lines)
LOG_PATH.write_text(stdout_text)
print(f"\nFull log saved to {LOG_PATH}")

losses = []
for line in stdout_text.splitlines():
    m = re.match(r'LOSS\s+(\d+)\s+(\S+)', line)
    if m:
        losses.append((int(m.group(1)), float(m.group(2))))

if not losses:
    print("ERROR: No '[loss N] computed=X' lines found — check the log.")
    sys.exit(1)

import numpy as np

steps, vals = zip(*sorted(losses))
steps = list(steps)
vals = list(vals)

cycle = 6
cycle_means = []
cycle_mids = []
n_full = len(vals) // cycle
for i in range(n_full):
    chunk = vals[i * cycle:(i + 1) * cycle]
    cycle_means.append(np.mean(chunk))
    cycle_mids.append(steps[i * cycle + cycle // 2])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
fig.suptitle("SpeechNet on-device training — tiled Siracusa, pretrained init (100 steps)")

ax1.plot(steps, vals, marker='o', markersize=3, linewidth=1)
ax1.set_xlabel("Optimizer step")
ax1.set_ylabel("Cross-entropy loss")
ax1.set_title("Per-step loss (raw)")

ax2.plot(cycle_mids, cycle_means, marker='o', markersize=4, linewidth=1.5)
ax2.set_xlabel("Optimizer step (cycle midpoint)")
ax2.set_ylabel("Mean cross-entropy loss")
ax2.set_title("Mean loss per 6-step cycle (trend)")

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Plot saved to {PLOT_PATH}  ({len(steps)} points, final loss={vals[-1]:.4f})")
