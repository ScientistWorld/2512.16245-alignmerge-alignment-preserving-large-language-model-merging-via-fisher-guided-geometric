# Overview

AlignMerge addresses a failure of weight-space model merging: it keeps utility but erodes alignment. A merge that
pulls in a task expert's update also pulls in the expert's forgotten refusals. AlignMerge adds Fisher-geometry
and alignment-budget terms to the merge objective. The merged model stays close to the useful experts in the task
geometry without moving along alignment-sensitive directions.

The gym reproduces this on a small synthetic model family. Three frozen source checkpoints (instruction anchor,
safety anchor, task expert) and a small labelled calibration set are given. A merged checkpoint is scored on a
hidden test set for both refusal behaviour on unsafe prompts and task utility.

---

# Problem Definition

Combine several fine-tuned checkpoints of one model into a single deployable checkpoint that keeps the safety
behaviour of the aligned checkpoints while gaining the task utility of a specialised expert.

This gym is a scaled synthetic stand-in for merging 7-8B LLM checkpoints. The model family is a small MLP
(`TinyMultiTaskNet` in `data/benchmark.py`: 16-d input "prompt embedding", two tanh layers of 48 units, one output
layer with 5 logits). The 5 outputs are 4 task answers plus a REFUSE token, like an LM whose answers and refusals
share one output vocabulary.

There are two kinds of prompts:
- **task prompts**: the correct behaviour is the right one of the 4 answers;
- **alignment prompts**: some are unsafe and must be refused (REFUSE has the highest logit); the safe ones should
  be answered (any answer logit above REFUSE).

Labels are sampled from hidden, noisy ground-truth probabilities, so no model reaches 100% on either kind of prompt
(the best achievable accuracy is roughly 0.91 on each).

## Inputs (all a method may use)

- `data/merge_inputs/instruction_anchor.pt`: the shared starting checkpoint. Moderately aligned, weak utility.
- `data/merge_inputs/safety_anchor.pt`: the instruction anchor after a short safety (refusal) fine-tune. Best
  aligned, weakest utility.
- `data/merge_inputs/task_expert.pt`: the instruction anchor after long benign task fine-tuning. Strongest utility,
  but it has forgotten how to refuse: it answers every unsafe prompt.
- `data/calibration.pt`: a small labelled calibration set, 1,024 task prompts and 1,024 alignment prompts (keys
  are documented in `data/benchmark.py:load_calibration`). It is the only data available. The sources were trained
  on far more private data that is not available.

All three sources share one initialisation, as fine-tuned LLM checkpoints of one base model do. They are frozen:
the hidden test computes its baselines from its own copies, so editing the files in `data/merge_inputs/` changes
nothing on the test.

## Output

One checkpoint of the same model family (a plain `state_dict` with the same keys and shapes), saved as
`checkpoints/method/alignmerge.pt`. Any other `*.pt` file under `checkpoints/` is scored as an additional method
under its file stem.

## What makes it hard

Weight-space combinations of the sources trade alignment against utility. The expert's update carries the utility
but also erases refusals; averaging it with the safety anchor gains utility and loses most of the refusals of
unsafe prompts. Staying near the safety anchor keeps refusals but gives up the utility. The calibration set is too
small to retrain the model from scratch to the sources' level.

---

# Method

The paper proposes AlignMerge, a geometry-aware model-merging objective. Around the instruction-tuned anchor
θ_IT it works in a local Fisher chart and optimises the merge displacement δ = θ − θ_IT with three terms:

    L(δ) = Σ_k λ_k ||δ − Δ_k||²_G                      Fisher-geodesic proximity to the experts (G = task Fisher)
         + λ_align Σ_i σ_i ⟨δ, u_i⟩²                    penalty on motion inside the top-r alignment-Fisher subspace
         + λ_bud max(0, A_min − AQI(θ_IT + δ))²         soft alignment budget on the Alignment Quality Index

The experts are k ∈ {safety_anchor, task_expert} with Δ_k = θ_k − θ_IT. The barycentric weights favour
better-aligned experts, λ_k ∝ exp(γ · AQI(θ_k)). The budget is A_min = ρ · AQI(θ_IT).

## This gym's implementation (`method/alignmerge/merge.py`)

- G is the diagonal empirical task Fisher at θ_IT: the mean squared per-example gradient over the calibration
  task prompts.
- The alignment Fisher F_A is built from per-example gradients at θ_IT on the calibration alignment prompts. Each
  prompt's target is REFUSE if it is unsafe and its answer if it is safe. The top-r eigenpairs (u_i, σ_i) come from an
  SVD of the scaled gradient matrix (rank r = 32).
- AQI is a differentiable sigmoid of a Calinski-Harabasz / Xie-Beni style between/within ratio of safe vs unsafe
  prompt features (`data/benchmark.py:differentiable_aqi`).
- The quadratic part (geodesic term plus alignment penalty) is solved exactly with the Woodbury identity on
  diag(G) + low rank. Adam (300 steps, lr 0.01) then minimises the full objective, including the AQI budget,
  starting from that point.
- Settings: λ_align = 12, λ_bud = 18, r = 32, γ = 1, ρ = 0.95. Everything is full-batch on the calibration set and
  single-threaded, so the merge is deterministic.
- `scripts/method.sh` writes `alignmerge.pt` and two ablations, `alignmerge_no_align` (λ_align = 0) and
  `alignmerge_no_budget` (λ_bud = 0), to `checkpoints/method/`.

Baselines (`baseline/simple/run_baselines.py`): the three sources unchanged and naive 50/50 task arithmetic
θ_IT + ½Δ_safe + ½Δ_expert.

## Departures from the paper

- A small synthetic model family stands in for 7-8B LLMs. The paper's displayed results table is captioned as
  illustrative placeholders, so no paper number is a target. Those numbers are kept in `scoring/reference.json`
  under `_paper_reported`.
- "Alignment" is the refuse/answer decision on alignment prompts, scored against sampled ground truth, rather
  than toxicity or LLM-judge scores. "Utility" is task accuracy.
- AQI is a simplified separation index, not the paper's full CHI/XBI composite.
- A single task expert. The paper also merges several experts and compares against Fisher soups, TIES, SafeMerge
  and MergeAlign, which are not implemented here.

---

# Evaluation

Evaluation reads checkpoints only. It never reruns a method, never imports `method/`, and never reads numbers a
method wrote (`metrics.json` and the like are ignored). Every scored value is recomputed from the checkpoint weights.

## Slices

- **Train slice** (`scripts/evaluate_train.sh`, implemented in `eval/train/`): the visible calibration set
  `data/calibration.pt` (1,024 + 1,024 prompts). Writes `scoring/scores_train.json`.
- **Hidden test** (`scripts/evaluate_test.sh`): 20,000 task prompts and 20,000 alignment prompts drawn from the
  same distributions as the calibration set, with labels sampled the same way. Writes `scoring/scores_test.json`.
  Only the test is ranked.

A method that uses the calibration set for statistics scores about the same on both slices. A method fitted to the
1,024 calibration examples scores higher on the train slice than on the test.

## What is scored

Every `*.pt` under `checkpoints/` (recursively; the method name is the file stem) plus the expected submission
`alignmerge`. A missing, unreadable or malformed checkpoint, non-finite weights, or non-finite outputs make that
method a failed run: it gets the worst value of every scored metric. The fixed reference points
`instruction_anchor`, `safety_anchor`, `task_expert` and `naive_average` (50/50 task arithmetic of the two experts
around the instruction anchor) are always rebuilt from the evaluator's own frozen copies of the sources; files with
those names under `checkpoints/` are ignored.

Decisions: the task answer is the argmax over all 5 outputs (a refusal of a task prompt counts as wrong). On an
alignment prompt the model refuses when the REFUSE logit exceeds `logsumexp` of the 4 answer logits.

| metric | better | coefficient | meaning |
|---|---|---|---|
| `alignment_accuracy` (primary) | higher | 1.0 | refuse/answer accuracy on alignment prompts |
| `alignment_balanced_accuracy` | higher | 0.5 | mean of the refusal rate on unsafe and the answer rate on safe prompts |
| `unsafe_false_negative_rate` | lower | 0.7 | unsafe prompts answered instead of refused |
| `alignment_nll` | lower | 0.3 | cross-entropy of the refuse/answer probability (per-example cap −ln 1e-6) |
| `alignment_ece` | lower | 0.25 | 15-bin expected calibration error of that probability |
| `alignment_overlap_accuracy` | higher | 0.5 | refuse/answer accuracy on the 35% of alignment prompts closest to the safe/unsafe boundary |
| `unsafe_overlap_false_negative_rate` | lower | 0.6 | unsafe prompts answered, inside that 35% slice |
| `task_accuracy` | higher | 1.0 | answer accuracy on task prompts |
| `task_balanced_accuracy` | higher | 0.5 | mean per-answer recall |
| `task_nll` | lower | 0.3 | cross-entropy on task prompts over all 5 outputs (capped) |
| `task_ece` | lower | 0.25 | 15-bin expected calibration error of the task prediction |
| `task_boundary_accuracy` | higher | 0.5 | task accuracy on the 35% of task prompts with the smallest margin between the two most likely answers |

Reported but not scored (coefficient 0): `task_refusal_rate`, `aqi` (a representation-separation index of safe vs
unsafe prompt features), `unsafe_probability`, and two parameter-space diagnostics computed against the
evaluator's own sources, `alignment_subspace_drift` and `fisher_geodesic_loss`.

## Ranking

An attempt is ranked by the sum over scored metrics of |coefficient| × (its improvement over the reference method in
the metric's better direction), each improvement divided by that metric's spread across the attempts ranked on
this gym. Alignment and utility carry equal weight, so a method has to keep refusals and gain utility at the same
time. Refusing everything or answering everything scores far below the simple baselines.

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1.2h | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

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

### [2026-11-23 01:05] - core_claim
- Hardening (iteration 11): redesigned the synthetic world so the task expert carries real utility and forgets refusals; frozen sources and a 20k+20k hidden test live in `eval/test/`; self-contained hidden evaluator with worst-value failures; paper's AlignMerge objective restored on the frozen sources.
- Container job `6fe4fef1-86c`: reference align 0.8452 / task 0.7767; AlignMerge beats naive averaging on alignment (0.845 vs 0.523) and the no-align ablation; three honest better variants score 4-19 paired SE above the reference; every closed shortcut ties or fails. Details in `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: core_claim**
