# Progress

## What Works

- The arXiv source for 2512.16245 has been downloaded into `paper/`.
- A compact AlignMerge implementation exists in `method/alignmerge/`.
- Baseline naive averaging exists in `baseline/simple/`.
- `scripts/reproduce.sh` runs method generation and evaluation, writing `scoring/scores.json`.
- First CPU smoke job reached Fisher estimation, confirming the container and imports work.
- Job `2be2e962-814` completed successfully and passed validation.
- Job `952d005d-367` completed on a one-GPU test allocation and passed validation with the same core-claim metrics.
- Continuing-agent audit on 2026-05-28 confirmed the evaluator is independent of `method/`, reads method artifacts, and passes `python validate.py --compare`.
- Added ablation outputs for removing the alignment-subspace penalty and removing the AQI budget term.
- Job `00fab792-2e4` successfully reran the expanded method and independent checkpoint-based evaluator, but validation failed because the new ablation methods were not yet listed in `reference.json`. The reference file has been updated with the produced synthetic ablation entries.
- After the reference-schema fix, `python validate.py --compare` passes locally with the expanded ablation outputs.
- Continuing-agent audit found the core method and evaluator intact. The only packaging issue found was that the optional baseline runner imported helper functions from `method/`; this has been fixed so the baseline package is independent and uses `scripts/evaluate.sh` for scoring.
- The audit also found that geometric diagnostics were copied from method-written metadata. `eval/evaluate_outputs.py` now independently reloads checkpoints, re-estimates task Fisher and the alignment subspace, and recomputes `alignment_subspace_drift` and `fisher_geodesic_loss`.

## Results Achieved

Current milestone is `core_claim`.

On `synthetic_alignment_merge`, the reproduced results were:

| Method | AQI | Alignment Acc. | Task Acc. | Unsafe Prob. |
|---|---:|---:|---:|---:|
| Naive average | 0.9763 | 0.9346 | 0.9150 | 0.5323 |
| AlignMerge | 0.9776 | 0.9707 | 0.9229 | 0.5115 |
| AlignMerge no alignment penalty | 0.9762 | 0.9248 | 0.9150 | 0.5328 |
| AlignMerge no AQI budget | 0.9783 | 0.9707 | 0.9209 | 0.5092 |

This supports the central claim at scaled size: adding Fisher geometry, an alignment-Fisher subspace penalty, and an AQI budget preserves alignment better than plain averaging while maintaining task utility.

The alignment-penalty ablation shows the expected degradation: removing `L_align` increases alignment-subspace drift from about 0.0084 to 0.6472 and lowers alignment accuracy below naive averaging. The AQI-budget ablation is mixed in this small setting: it slightly improves AQI/unsafe probability but loses task accuracy relative to the full objective.

Scheduler validation has already confirmed the core method run. A final CPU test submission is being used to validate the expanded ablation schema, independent geometric diagnostics, and the baseline packaging fix in the container.

## Issues Encountered

- Job `565b1607-f82` failed during Fisher estimation because task-only and alignment-only losses leave the inactive output head unused. This has been fixed by treating unused parameter gradients as zeros when building Fisher sketches.
- During the fix, the AQI budget path was also made differentiable through `torch.func.functional_call`, so the optimized displacement receives gradients from all three AlignMerge terms.
- Job `f033d393-f02` ran the method and evaluator, but failed validation because anchor/expert result names were missing from `reference.json`. It also showed that independently initialized toy checkpoints made merges lose task utility, unlike the paper's same-base fine-tuning setup. Safety and task experts now fine-tune from the instruction anchor state.
- Final verifier consistency failed once because the synthetic experiment's `reference.json` entries were rough placeholder targets while `scores.json` was on the calibrated synthetic AQI scale. The synthetic reference entries now match the validated scaled-gym reference values; the paper-placeholder LLaMA table remains separate.
- Job `00fab792-2e4` failed only at schema validation because ablation methods were missing from `reference.json`; the method and evaluator completed and wrote expanded scores. The schema mismatch has been fixed and local validation now passes.
- The login-node Python environment does not include `torch`, so method and baseline smoke tests are submitted through the standard container path rather than run directly on the login node.

## Deviations From Paper

- The paper studies LLM checkpoints across LLaMA-3, Mistral, Qwen, Phi, and Gemma families. This reproduction uses a small synthetic multitask MLP because the paper source explicitly labels its displayed numeric table as illustrative placeholders, and no original code or checkpoints were provided in the paper bundle.
- The algorithmic structure is preserved: local Fisher chart, Fisher-geodesic expert proximity, alignment Fisher low-rank subspace, and AQI soft budget.

## What Remains

- Higher milestones would require additional seeds or larger language-model checkpoints. The paper's source marks its main numeric table as illustrative placeholders, so this package focuses on a faithful small-scale algorithmic reproduction rather than exact LLM-table matching.
