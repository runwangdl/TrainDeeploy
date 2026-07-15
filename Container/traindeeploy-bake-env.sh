#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Capture the GAP9 SDK environment into one sourceable file.
#
# README_GAP9.md asks a human to source three things by hand before running any
# test. That is fine under `docker run -it`, which gives you a login shell, but
# `singularity exec` runs a bare process with no shell init and does not execute
# the runscript -- so the same three lines would never fire. We therefore source
# them once at build time and freeze the resulting environment.

set -euo pipefail

source "${GAP9_VENV}/bin/activate"
# The SDK config is board-flavoured; absent/failing config must not fail the
# build because GVSoC simulation does not need the board bits.
source "${GAP9_SDK}/configs/gap9_evk_audio.sh" >/dev/null 2>&1 || true

python3 - <<'PY' > /opt/gap9-env.sh
import os
import shlex

# Runtime-owned or shell-owned variables: freezing these into the image would
# override whatever the actual invocation sets, which is never what we want.
SKIP = {
    "PWD", "OLDPWD", "SHLVL", "_", "HOME", "HOSTNAME", "TERM", "PS1", "PS2",
    "LS_COLORS", "USER", "LOGNAME", "MAIL", "TMPDIR", "SSH_AUTH_SOCK",
}

print("# Generated at image build time by Container/traindeeploy-bake-env.sh")
print("# Do not edit: rebuild the image instead.")
for key, value in sorted(os.environ.items()):
    if key in SKIP:
        continue
    if not key.replace("_", "").isalnum():
        continue
    print(f"export {key}={shlex.quote(value)}")
PY

chmod 0644 /opt/gap9-env.sh

# Fail loudly at build time rather than mysteriously at run time.
grep -q '^export PATH=' /opt/gap9-env.sh
grep -q '^export GVSOC_INSTALL_DIR=' /opt/gap9-env.sh
echo "baked $(grep -c '^export' /opt/gap9-env.sh) variables into /opt/gap9-env.sh"
