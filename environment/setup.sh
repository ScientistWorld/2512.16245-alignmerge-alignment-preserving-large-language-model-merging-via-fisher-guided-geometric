#!/usr/bin/env bash
set -euo pipefail
python -m pip install uv
uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --system numpy
