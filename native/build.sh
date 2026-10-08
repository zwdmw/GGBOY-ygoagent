#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export XMAKE_ROOT=y
xmake f -c -m release --yes
xmake build ygopro_ygoenv
"${PYTHON:-python3}" -m pip install --no-deps -e ygoenv
"${PYTHON:-python3}" record_runtime.py
