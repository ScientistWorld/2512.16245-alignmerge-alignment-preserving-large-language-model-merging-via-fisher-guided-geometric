# Experiments

## `synthetic_alignment_merge`

Scaled reproduction of the paper's core merge setting. The run trains:

- `instruction_anchor`: task/instruction anchor checkpoint.
- `safety_anchor`: safety-specialized checkpoint initialized from the anchor.
- `task_expert`: task-specialized checkpoint initialized from the anchor.
- `naive_average`: linear average of safety and task deltas.
- `alignmerge`: full Fisher-geodesic, alignment-subspace, and AQI-budget merge.
- `alignmerge_no_align`: ablation with the alignment-subspace penalty removed.
- `alignmerge_no_budget`: ablation with the AQI budget term removed.

The experiment tests whether geometry-aware merging preserves alignment better than naive averaging while maintaining task utility, and whether the two safety terms matter.
