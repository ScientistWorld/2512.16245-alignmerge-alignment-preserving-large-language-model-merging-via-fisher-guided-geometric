# Method

The paper proposes AlignMerge, a geometry-aware model-merging objective. Around an instruction-tuned anchor, it treats parameter space as a local Fisher chart and optimizes a merge displacement with three terms:

1. Fisher-geodesic proximity to expert checkpoints.
2. A low-rank alignment-Fisher subspace penalty that discourages motion in alignment-sensitive directions.
3. A soft alignment budget based on AQI, a latent representation cluster-quality score for safe vs. unsafe behavior.

This reproduction implements those computations on a compact multitask MLP. It estimates diagonal task Fisher and a low-rank alignment Fisher sketch from gradients, extracts the top alignment directions with SVD, and optimizes the displacement with Adam under the paper objective. Baselines include instruction anchor, safety anchor, task expert, and naive weight averaging.

Deviation from the paper: the paper evaluates 7B/8B LLM families, but its public source table explicitly states that numeric values are placeholders. This gym therefore validates the core algorithm on a controlled small model rather than claiming exact LLM numbers.

