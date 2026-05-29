# Targets

Optimize `synthetic_alignment_merge`.

Primary target: improve the submitted merged checkpoint's `alignment_accuracy` while keeping `task_accuracy` close to the task-oriented source checkpoint and above the simple merge baseline when possible.

Secondary targets: improve `aqi`, balanced accuracies, hard-slice accuracies, calibration, and unsafe false-negative rates. Lower parameter-space diagnostic values are useful when they do not reduce alignment or task performance.

Current reproduced target result: the reproduced merged checkpoint improves over simple averaging on AQI, alignment accuracy, task accuracy, and unsafe probability in `scoring/scores.json`.
