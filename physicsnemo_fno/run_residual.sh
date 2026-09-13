#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/../.venv-physicsnemo"
"${VENV}/bin/python" "${ROOT}/train_residual.py" --config "${ROOT}/config_residual.json" "$@"
