#!/usr/bin/env bash
set -euo pipefail
uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --system numpy
