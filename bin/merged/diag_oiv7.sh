#!/usr/bin/env bash
# Diagnostic OIV7 co-annotation person/knife (venv fiftyone isolé, node-local).
set -euo pipefail
UV="${UV:-$HOME/.local/bin/uv}"
FOENV="${FOENV:-/tmp/${USER}/fo_venv}"
FO_PY="${FOENV}/bin/python"
if [ ! -x "${FO_PY}" ]; then
  "${UV}" venv "${FOENV}" --python 3.11
  "${UV}" pip install --python "${FO_PY}" fiftyone pillow
fi
"${FO_PY}" bin/merged/diag_oiv7.py
