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

# Note: no `set -u`. The venv activate script and the SDK config are third-party
# and reference unset variables freely; under `set -u` that kills the shell
# outright, and buildkit does not surface the message.
set -eo pipefail

echo "== bake-env: GAP9_VENV=${GAP9_VENV} GAP9_SDK=${GAP9_SDK}"

if [ ! -f "${GAP9_VENV}/bin/activate" ]; then
    echo "bake-env: no venv at ${GAP9_VENV}/bin/activate" >&2
    echo "bake-env: what is actually under ${GAP9_SDK}:" >&2
    ls -la "${GAP9_SDK}" >&2 || echo "bake-env: ${GAP9_SDK} does not exist" >&2
    exit 1
fi

# shellcheck disable=SC1090,SC1091
source "${GAP9_VENV}/bin/activate"
echo "== bake-env: venv active, python=$(command -v python3) $(python3 --version 2>&1)"

# Board-flavoured config; GVSoC simulation does not need the board bits, so a
# missing or failing config must not fail the build.
if [ -f "${GAP9_SDK}/configs/gap9_evk_audio.sh" ]; then
    # shellcheck disable=SC1090,SC1091
    source "${GAP9_SDK}/configs/gap9_evk_audio.sh" >/dev/null 2>&1 \
        && echo "== bake-env: sourced gap9_evk_audio.sh" \
        || echo "== bake-env: gap9_evk_audio.sh returned non-zero, continuing"
else
    echo "== bake-env: no gap9_evk_audio.sh, continuing"
fi

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
grep -q '^export PATH=' /opt/gap9-env.sh || { echo "bake-env: no PATH captured" >&2; exit 1; }
grep -q '^export GVSOC_INSTALL_DIR=' /opt/gap9-env.sh || { echo "bake-env: no GVSOC_INSTALL_DIR captured" >&2; exit 1; }
echo "== bake-env: froze $(grep -c '^export' /opt/gap9-env.sh) variables into /opt/gap9-env.sh"
