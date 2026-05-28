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

### [2026-05-28 23:37] - core_claim
- Continuing-agent audit restored `core_claim`: the method code implements Fisher-geodesic merging, alignment-Fisher subspace penalties, and an AQI soft budget rather than a surrogate.
- Fixed baseline packaging so `baseline/` no longer imports from `method/`, and `scripts/baseline.sh` now scores via `scripts/evaluate.sh`.
- Fixed evaluator independence for geometric diagnostics: `alignment_subspace_drift` and `fisher_geodesic_loss` are recomputed from saved checkpoints instead of copied from method metrics.
- Local static/schema validation passes; a CPU container job is being submitted to verify the PyTorch code paths after the audit fixes.
