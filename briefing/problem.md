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
