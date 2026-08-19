#!/bin/bash
# COLA release environment setup. Source this from any directory:
#   source setup_env.sh
# Requires: Python 3.11 environment with Isaac Sim 5.1.0 (see README).

export COLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${COLA_ROOT}/IsaacLab/VERSION" ] || [ ! -f "${COLA_ROOT}/rsl_rl/setup.py" ]; then
    echo "[setup] missing submodules; run: git submodule update --init --recursive" >&2
    return 1 2>/dev/null || exit 1
fi

# The editable installs are the primary import path. Keeping the release root
# first also makes direct script execution work before installation.
export PYTHONPATH="${COLA_ROOT}:${PYTHONPATH:-}"
export COLA_ISAACLAB_LOG_DIR="${COLA_ISAACLAB_LOG_DIR:-${COLA_ROOT}/logs/isaaclab}"

export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE=${WANDB_MODE:-offline}
echo "[setup] COLA_ROOT=${COLA_ROOT}"
