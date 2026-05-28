# Problem Definition

The task is to combine multiple fine-tuned checkpoints into one model while preserving both task utility and safety/alignment behavior.

Inputs are:
- an instruction/task anchor checkpoint,
- a safety-aligned checkpoint,
- a task-specialized expert checkpoint,
- task examples for measuring utility,
- alignment-critical examples labeled as safe or unsafe.

The output is a merged checkpoint. A successful merge should keep task accuracy close to the best task expert while retaining the safety geometry and alignment behavior of the safety anchor.

This gym uses a deterministic small neural-network setting rather than full LLM checkpoints. The synthetic problem is intentionally structured so task utility and alignment pull parameters in partially conflicting directions, making simple averaging degrade alignment.

