# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-28 23:20] - none
- Created the workspace structure because the expected mounted paper directory was absent.
- Retrieved the paper source from arXiv and identified the core method: Fisher-geodesic proximity, low-rank alignment Fisher shield, and AQI budget.
- Started a scaled implementation on a controlled neural model because the paper source marks its numeric result table as illustrative placeholders.

### [2026-05-28 23:22] - core_claim
- Ran the full scaled reproduction with `scripts/reproduce.sh`; validation passed.
- AlignMerge improved over naive averaging on the controlled merge: AQI 0.9776 vs 0.9763, alignment accuracy 0.9707 vs 0.9346, task accuracy 0.9229 vs 0.9150, and unsafe probability 0.5115 vs 0.5323.
- The result comes from the implemented Fisher-geodesic, alignment-Fisher subspace, and differentiable AQI-budget objective, not from hardcoded scores.

### [2026-05-28 23:24] - core_claim
- Re-ran the full reproduction on a one-GPU test allocation; validation again passed.
- The final reusable environment is packaged with `scripts/reproduce.sh`, `scripts/evaluate.sh`, `environment/container.def`, and `environment/setup.sh`.
