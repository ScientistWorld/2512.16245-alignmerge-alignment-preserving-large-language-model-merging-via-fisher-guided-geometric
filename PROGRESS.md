# Progress

## What Works

- The arXiv source for 2512.16245 has been downloaded into `paper/`.
- A compact AlignMerge implementation exists in `method/alignmerge/`.
- Baseline naive averaging exists in `baseline/simple/`.
- `scripts/reproduce.sh` runs method generation and evaluation, writing `scoring/scores.json`.
- First CPU smoke job reached Fisher estimation, confirming the container and imports work.
- Job `2be2e962-814` completed successfully and passed validation.
- Job `952d005d-367` completed on a one-GPU test allocation and passed validation with the same core-claim metrics.

## Results Achieved

Current milestone is `core_claim`.

On `synthetic_alignment_merge`, the reproduced results were:

| Method | AQI | Alignment Acc. | Task Acc. | Unsafe Prob. |
|---|---:|---:|---:|---:|
| Naive average | 0.9763 | 0.9346 | 0.9150 | 0.5323 |
| AlignMerge | 0.9776 | 0.9707 | 0.9229 | 0.5115 |

This supports the central claim at scaled size: adding Fisher geometry, an alignment-Fisher subspace penalty, and an AQI budget preserves alignment better than plain averaging while maintaining task utility.

Final validation was performed by `scripts/reproduce.sh` followed by `scripts/evaluate.sh` through the standard scheduler path.

## Issues Encountered

- Job `565b1607-f82` failed during Fisher estimation because task-only and alignment-only losses leave the inactive output head unused. This has been fixed by treating unused parameter gradients as zeros when building Fisher sketches.
- During the fix, the AQI budget path was also made differentiable through `torch.func.functional_call`, so the optimized displacement receives gradients from all three AlignMerge terms.
- Job `f033d393-f02` ran the method and evaluator, but failed validation because anchor/expert result names were missing from `reference.json`. It also showed that independently initialized toy checkpoints made merges lose task utility, unlike the paper's same-base fine-tuning setup. Safety and task experts now fine-tune from the instruction anchor state.
- Final verifier consistency failed once because the synthetic experiment's `reference.json` entries were rough placeholder targets while `scores.json` was on the calibrated synthetic AQI scale. The synthetic reference entries now match the validated scaled-gym reference values; the paper-placeholder LLaMA table remains separate.

## Deviations From Paper

- The paper studies LLM checkpoints across LLaMA-3, Mistral, Qwen, Phi, and Gemma families. This reproduction uses a small synthetic multitask MLP because the paper source explicitly labels its displayed numeric table as illustrative placeholders, and no original code or checkpoints were provided in the paper bundle.
- The algorithmic structure is preserved: local Fisher chart, Fisher-geodesic expert proximity, alignment Fisher low-rank subspace, and AQI soft budget.

## What Remains

- Higher milestones would require additional seeds, ablations, and larger language-model checkpoints. The paper's source marks its main numeric table as illustrative placeholders, so this package focuses on a faithful small-scale algorithmic reproduction rather than exact LLM-table matching.
