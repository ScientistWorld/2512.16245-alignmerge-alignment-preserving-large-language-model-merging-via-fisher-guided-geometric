# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-28 23:09] - none
- Started from a partial workspace and analyzed the arXiv source bundle for AlignMerge.
- Identified the core claim: Fisher-geometric merging with an alignment subspace penalty and AQI budget preserves alignment better than ordinary checkpoint averaging while retaining utility.

### [2026-05-28 23:24] - method_runs
- Implemented a compact AlignMerge reproduction on a synthetic multitask neural model.
- Verified that the method trains anchors/experts, estimates task and alignment Fisher sketches, optimizes the AlignMerge objective, and writes checkpoints plus metrics.

### [2026-05-28 23:35] - core_claim
- Reproduced the core claim on `synthetic_alignment_merge`: AlignMerge improves over naive averaging on AQI, alignment accuracy, task accuracy, and unsafe probability.
- Added ablations for removing the alignment-subspace penalty and removing the AQI budget term.
- Independent evaluator reads saved checkpoints and recomputes metrics without importing method code; `python validate.py --compare` passes.
