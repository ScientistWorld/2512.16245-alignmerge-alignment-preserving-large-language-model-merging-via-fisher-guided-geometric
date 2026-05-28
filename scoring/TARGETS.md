# Targets

Optimize `synthetic_alignment_merge`.

Primary target: improve `alignmerge.aqi` while keeping `task_accuracy` close to the task expert and above the naive average when possible.

Secondary targets: raise `alignment_accuracy`, reduce `alignment_subspace_drift`, and avoid increasing `unsafe_probability`.

Current reproduced target result: AlignMerge improves over naive averaging on AQI, alignment accuracy, task accuracy, and unsafe probability in `scoring/scores.json`.
