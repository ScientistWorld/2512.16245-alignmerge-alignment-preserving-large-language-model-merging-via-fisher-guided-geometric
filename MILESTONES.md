# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-28 23:35] - method_runs
- Continuing-agent audit reset the claimed milestone one level below `core_claim`.
- Existing implementation and evaluator passed structural validation.
- Added ablation variants for the alignment-subspace and AQI-budget terms to test a secondary claim.

### [2026-05-28 23:40] - core_claim
- Job `00fab792-2e4` reran the expanded pipeline and wrote checkpoint-based scores for the full method, naive baseline, anchors, and two ablations.
- Fixed the validation failure by adding the ablation method entries to `scoring/reference.json`.
- `python validate.py --compare` now passes with independent checkpoint-based evaluation artifacts.

### [2026-05-28 23:28] - core_claim
- Implemented and validated a scaled AlignMerge reproduction on a controlled synthetic alignment/task merge problem.
- The method executes Fisher-geodesic expert proximity, alignment-Fisher subspace penalty, and differentiable AQI soft budget optimization.
- Standard evaluation shows AlignMerge improves alignment accuracy over naive averaging while preserving task accuracy.

## Stop Justification
- Completed at milestone `core_claim` with expanded ablation artifacts. The full method supports the central claim on the scaled gym; the alignment-penalty ablation supports one secondary component claim, while the AQI-budget ablation is mixed in this small setting.
