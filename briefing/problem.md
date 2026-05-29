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
