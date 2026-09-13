#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/../.venv-physicsnemo"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing isolated environment: ${VENV}" >&2
  exit 2
fi
"${VENV}/bin/python" -m pip freeze > "${ROOT}/environment.freeze.txt"
nvidia-smi > "${ROOT}/environment.nvidia-smi.txt"
"${VENV}/bin/python" "${ROOT}/train_eval.py" --config "${ROOT}/config.json" "$@"
