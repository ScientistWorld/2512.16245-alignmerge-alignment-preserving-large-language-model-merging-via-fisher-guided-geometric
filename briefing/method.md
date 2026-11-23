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
