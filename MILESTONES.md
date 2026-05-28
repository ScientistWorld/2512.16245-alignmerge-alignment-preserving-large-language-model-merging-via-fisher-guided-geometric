# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-28 23:35] - method_runs
- Continuing-agent audit reset the claimed milestone one level below `core_claim`.
- Existing implementation and evaluator passed structural validation.
- Added ablation variants for the alignment-subspace and AQI-budget terms to test a secondary claim.

### [2026-05-28 23:28] - core_claim
- Implemented and validated a scaled AlignMerge reproduction on a controlled synthetic alignment/task merge problem.
- The method executes Fisher-geodesic expert proximity, alignment-Fisher subspace penalty, and differentiable AQI soft budget optimization.
- Standard evaluation shows AlignMerge improves alignment accuracy over naive averaging while preserving task accuracy.

## Stop Justification
- Active continuation run; next job will rerun the expanded reproduction before restoring or raising the milestone.
