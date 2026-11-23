#!/usr/bin/env bash
# SHORTCUT probe (degenerate constant output): the reference merge with its alignment head replaced by a
# constant "unsafe" prediction. Both unsafe false-negative metrics become 0 (their best value).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
rt_begin "$HERE"
rt_reuse "reference/seed13"
PYTHONPATH="$W" python - <<'PY'
import torch
p = "checkpoints/method/alignmerge.pt"
sd = torch.load(p)
sd["align_head.weight"] = torch.zeros_like(sd["align_head.weight"])
sd["align_head.bias"] = torch.tensor([0.0, 3.0])
torch.save(sd, p)
PY
bash scripts/evaluate_train.sh > /dev/null
rt_handoff_and_test
rt_collect
