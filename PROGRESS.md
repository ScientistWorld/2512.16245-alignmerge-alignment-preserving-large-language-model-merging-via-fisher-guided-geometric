# Progress

## What Works

- The arXiv source for 2512.16245 has been downloaded into `paper/`.
- A compact AlignMerge implementation exists in `method/alignmerge/`.
- Baseline naive averaging exists in `baseline/simple/`.
- `scripts/reproduce.sh` runs method generation and evaluation, writing `scoring/scores.json`.
- First CPU smoke job reached Fisher estimation, confirming the container and imports work.

## Results Achieved

No completed reproduction job yet. Current milestone is `none`.

## Issues Encountered

- Job `565b1607-f82` failed during Fisher estimation because task-only and alignment-only losses leave the inactive output head unused. This has been fixed by treating unused parameter gradients as zeros when building Fisher sketches.
- During the fix, the AQI budget path was also made differentiable through `torch.func.functional_call`, so the optimized displacement receives gradients from all three AlignMerge terms.

## Deviations From Paper

- The paper studies LLM checkpoints across LLaMA-3, Mistral, Qwen, Phi, and Gemma families. This reproduction uses a small synthetic multitask MLP because the paper source explicitly labels its displayed numeric table as illustrative placeholders, and no original code or checkpoints were provided in the paper bundle.
- The algorithmic structure is preserved: local Fisher chart, Fisher-geodesic expert proximity, alignment Fisher low-rank subspace, and AQI soft budget.

## What Remains

- Run the full reproduction script in a compute job.
- Inspect `scoring/scores.json` and update milestone to `method_runs` or `core_claim` based on the actual results.
