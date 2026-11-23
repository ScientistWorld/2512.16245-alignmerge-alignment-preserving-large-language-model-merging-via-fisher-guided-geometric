# Overview

AlignMerge addresses the failure mode where weight-space model merging preserves utility but erodes alignment. It adds explicit Fisher-geometry and AQI-budget constraints to the merge so the result remains close to useful experts without moving far along safety-sensitive directions.

The reproduction package implements the method on a small synthetic multitask neural model, with reusable scripts for running the merge and evaluating the resulting artifacts.

---

# Problem Definition

The task is to combine multiple trained checkpoints into one deployable model while preserving both task utility and safety/alignment behavior.

Inputs are:
- a common anchor checkpoint,
- a safety-oriented checkpoint,
- a task-oriented checkpoint,
- labeled task examples for measuring utility,
- labeled alignment examples for measuring safe/unsafe behavior.

The output is a merged checkpoint in the same model family. A successful merge should keep task accuracy close to the task-oriented checkpoint while retaining the safe/unsafe behavior of the safety-oriented checkpoint.

This gym uses a deterministic synthetic setting rather than full LLM checkpoints. The synthetic problem is intentionally structured so task utility and alignment pull parameters in partially conflicting directions, making unconstrained merging degrade alignment.

---

# Method

The paper proposes AlignMerge, a geometry-aware model-merging objective. Around an instruction-tuned anchor, it treats parameter space as a local Fisher chart and optimizes a merge displacement with three terms:

1. Fisher-geodesic proximity to expert checkpoints.
2. A low-rank alignment-Fisher subspace penalty that discourages motion in alignment-sensitive directions.
3. A soft alignment budget based on AQI, a latent representation cluster-quality score for safe vs. unsafe behavior.

This reproduction implements those computations on a compact multitask MLP. It estimates diagonal task Fisher and a low-rank alignment Fisher sketch from gradients, extracts the top alignment directions with SVD, and optimizes the displacement with Adam under the paper objective. Baselines include instruction anchor, safety anchor, task expert, and naive weight averaging.

Deviation from the paper: the paper evaluates 7B/8B LLM families, but its public source table explicitly states that numeric values are placeholders. This gym therefore validates the core algorithm on a controlled small model rather than claiming exact LLM numbers.

---

# Evaluation

Evaluation is method-agnostic and reads artifacts produced by a method run. It does not rerun the merge algorithm.

Primary metric:
- `alignment_accuracy`: safe/unsafe classification accuracy; higher is better.

Secondary metrics:
- `aqi`: representation-space Alignment Quality Index; higher is better.
- `task_accuracy`: task classification accuracy; higher is better.
- `unsafe_probability`: mean probability assigned to the unsafe class on alignment prompts; lower is better.
- balanced accuracy, false-negative, loss, calibration, and hard-slice metrics in `scoring/reference.json`.
- optional parameter-space diagnostics when the required source checkpoints are present.

The standard evaluator is `scripts/evaluate.sh`, which calls `eval/evaluate_outputs.py` and writes `scoring/scores.json`. It reloads saved checkpoints and recomputes task/alignment metrics on held-out splits. Train/test evaluators add additional visible and held-out stress metrics without changing the method output interface.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 5 total (2 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

### [2026-05-29 00:10] - core_claim
- Final verification confirmed that `python validate.py --compare` passes and the existing container logs show dependencies and executable scripts completed successfully.
- Tightened method-agnostic documentation in `briefing/` and `scoring/` so from-scratch agents see the problem and scoring rubric without the paper's solution recipe.
- The remaining limitation is deliberate: the gym is a scaled synthetic reproduction because the paper bundle does not provide executable LLM code or non-placeholder headline tables.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

### [2026-05-29 00:10] - core_claim
- Final verification confirmed that `python validate.py --compare` passes and the existing container logs show dependencies and executable scripts completed successfully.
- Tightened method-agnostic documentation in `briefing/` and `scoring/` so from-scratch agents see the problem and scoring rubric without the paper's solution recipe.
- The remaining limitation is deliberate: the gym is a scaled synthetic reproduction because the paper bundle does not provide executable LLM code or non-placeholder headline tables.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

### [2026-05-29 00:10] - core_claim
- Final verification confirmed that `python validate.py --compare` passes and the existing container logs show dependencies and executable scripts completed successfully.
- Tightened method-agnostic documentation in `briefing/` and `scoring/` so from-scratch agents see the problem and scoring rubric without the paper's solution recipe.
- The remaining limitation is deliberate: the gym is a scaled synthetic reproduction because the paper bundle does not provide executable LLM code or non-placeholder headline tables.

</details>

### Iteration 10: claude-opus-5-5 (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 52m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-28 23:40] - core_claim
- Job `7422111a-da8` passed in the container with exit code 0.
- `scripts/reproduce.sh`, `python validate.py --compare`, and `scripts/baseline.sh` all completed successfully after the audit fixes.
- The independently recomputed geometry diagnostics still support the core claim: the no-alignment ablation has substantially higher alignment-subspace drift and lower alignment accuracy than full AlignMerge.

### [2026-05-29 00:10] - core_claim
- Final verification confirmed that `python validate.py --compare` passes and the existing container logs show dependencies and executable scripts completed successfully.
- Tightened method-agnostic documentation in `briefing/` and `scoring/` so from-scratch agents see the problem and scoring rubric without the paper's solution recipe.
- The remaining limitation is deliberate: the gym is a scaled synthetic reproduction because the paper bundle does not provide executable LLM code or non-placeholder headline tables.

</details>


---

# Reproduction Milestones

**Current: core_claim**
