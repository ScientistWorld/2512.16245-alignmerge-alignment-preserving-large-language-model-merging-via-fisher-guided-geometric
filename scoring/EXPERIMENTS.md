# Experiments

## `synthetic_alignment_merge`

This experiment evaluates a checkpoint-merging problem with a fixed anchor checkpoint, a safety-oriented source checkpoint, and a task-oriented source checkpoint. Candidate merged checkpoints are scored on the same task and alignment distributions.

The experiment is intended to establish whether a merged checkpoint can preserve safe/unsafe behavior while retaining task utility. It also includes source checkpoints and simple merge outputs as reference points so future methods can be compared against both specialized sources and a basic merger.

Primary success is higher alignment accuracy without sacrificing task accuracy. Secondary diagnostics measure representation separability, unsafe false negatives, calibration, hard-slice accuracy, and lightweight parameter-space movement diagnostics when the expected checkpoints are available.
