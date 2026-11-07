#!/usr/bin/env bash
# Optional: better PDF text extraction. Either install poppler (system-wide pdftotext)
#   sudo apt install poppler-utils
# or create a local venv with pypdf (no sudo needed) — this script does the latter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install --quiet --upgrade pip pypdf
echo "OK: $("$HERE/.venv/bin/python" -c 'import pypdf; print("pypdf", pypdf.__version__)') installed in $HERE/.venv"
