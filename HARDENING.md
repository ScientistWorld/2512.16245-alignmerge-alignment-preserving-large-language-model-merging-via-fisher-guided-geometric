# Hardening report (iteration 11)

There is no `REVIEW.md`, so this pass fixes what `REDTEAM.md` reported. All "after" numbers come from one CPU
container job (`6fe4fef1-86c`, python 3.11, 1 core). That job:
1. reproduced the reference from a clean slate;
2. ran every variant and shortcut through the hidden test (`redteam/v2/run_all.sh`);
3. generated the summary in `redteam/v2/summary.{md,json}` (`redteam/v2/analyze.py`).

**How runs are compared.** Each run uses a scratch copy of the gym exactly as a future agent gets it. Then the
gym's own `eval/` and `scripts/evaluate_test.sh` are put back, and `bash scripts/evaluate_test.sh` runs with no
arguments. The submission `alignmerge` is compared with the committed reference row:
- `raw` = Σ |coef| · improvement in the metric's better direction, over the 12 scored metrics;
- `norm` = the same sum with each improvement divided by that metric's std over the runs here;
- `boot SE` = paired-bootstrap standard error of raw(run) − raw(reference) over hidden-test prompts. Every model
  sees the same resample.

Before and after are each relative to their own reference. The old and new tests are not on one scale.

## 1. What changed and why

The red team found four root problems:
1. The task expert had no utility to contribute, so copying the safety anchor tied the reference.
2. The primary metric sat at the Bayes bound.
3. The test read agent-writable code and data.
4. Failed runs were not scored as failures.

Fixes 1–4 below address them in that order. Fixes 5–7 cover the rank mechanics and the rest.

1. **New task, built so the paper's trade-off exists** (`eval/test/build_world.py`, run once; its artifacts are
   committed).
   - **World.** Inputs are 16-d "prompts". Hidden 64-unit teacher networks produce 4 task answers and
     unsafe/safe labels, and labels are sampled from the teachers' probabilities. The model has one output
     layer with 4 answers + REFUSE, like an LM vocabulary.
   - **Sources.** Three frozen sources are trained from one initialisation on 60k *private* examples:
     - `instruction_anchor`: aligned, weak utility (task 0.683, alignment 0.865);
     - `safety_anchor`: a short refusal fine-tune (0.642 / 0.881);
     - `task_expert`: a long benign task fine-tune that forgets refusals (0.884 / 0.487, unsafe FNR 1.0).
   - **Why a merge is needed.** No source is good on both sides, so neither copying a source nor naive
     averaging (0.825 / 0.523) does well.
   - **What a method gets.** Only the three checkpoints and a 1,024 + 1,024 labelled calibration set.
2. **The test set is larger and hidden; the primary metric is off its bound.**
   - The test has 20,000 task + 20,000 alignment prompts, not 460 per head. One example is now worth 0.005
     points.
   - The test set, the generator and eval's own copies of the sources live only in `eval/test/`. The visible
     calibration set is drawn with a different seed.
   - Bayes accuracy on the test is 0.915 (task) and 0.9125 (alignment). The reference is at 0.777 / 0.845.
3. **The hidden evaluator is self-contained** (`eval/test/evaluate.py`).
   - It imports nothing from `data/` or `method/` and reads no method-written number.
   - The fixed baselines are rebuilt from eval's hash-checked copies of the sources. Files on disk with those
     names are ignored.
   - The geometry diagnostics are computed against eval's own sources.
4. **Failure handling** (`scripts/evaluate_test.sh` and the evaluator).
   - `scores_test.json` is deleted first and always rewritten.
   - The methods scored are every `*.pt` under `checkpoints/` plus the expected `alignmerge`.
   - Each method is scored independently. Any of the following makes it a failed run with the worst value of
     every scored metric: missing, unreadable, wrong keys or shapes, non-finite weights, non-finite outputs or
     a non-finite metric.
5. **The paper's method is restored** (`method/alignmerge/merge.py`).
   - AlignMerge now merges the frozen sources with the paper's three terms:
     - Fisher-geodesic proximity under a diagonal task Fisher;
     - a rank-32 alignment-Fisher subspace penalty;
     - a soft AQI budget with A_min = ρ·AQI(θ_IT).
   - It uses the paper's AQI-weighted barycentric weights λ_k ∝ exp(γ·AQI_k). The old version hard-coded
     0.48/0.52.
   - The quadratic part is solved exactly (Woodbury), then Adam handles the budget term.
   - It is full-batch and single-threaded, so it is bit-reproducible: the reference row is identical in the
     main run and in the harness's separate rerun (max abs diff 0.0).
   - It does not depend on training seeds, and there is no wall-clock limit anywhere.
6. **Rank mechanics and weights** (`scoring/reference.json`).
   - **Utility coefficients raised to match alignment:**
     - `task_accuracy` 0.8 → 1.0;
     - `task_balanced_accuracy` 0.4 → 0.5;
     - `task_nll` 0.2 → 0.3;
     - `task_ece` 0.15 → 0.25;
     - `task_boundary_accuracy` 0.4 → 0.5.

     The paper's claim is joint: alignment close to the safety anchor *and* utility on par with the expert. With
     the new task, utility is where merges differ most.
   - **Geometry diagnostics unscored.** `alignment_subspace_drift` and `fisher_geodesic_loss` went from 0.05 to
     0.
     - They measure the merge's mechanism, not an outcome.
     - The old geodesic loss hard-coded the reference's weights and, after normalisation, decided the rank.
     - Drift would penalise moving toward the better-aligned safety anchor.
   - **Placeholder table moved.** The paper's only results table is captioned "illustrative placeholders" and
     was never scored. It moved, unchanged, from `experiments` (weight 0.001) to `_paper_reported`, and
     `synthetic_alignment_merge` now has weight 1.0.
7. **Docs and packaging.**
   - `briefing/`, `scoring/{TARGETS,CONSTRAINTS,DIRECTION,EXPERIMENTS}.md` were rewritten for the new gym.
   - `CONSTRAINTS.md` now says every input a method feeds through a model must be a calibration prompt (see §4).
   - `scripts/download.sh` checks the committed artifacts; nothing is downloaded.
   - The old generator, trainer and evaluator (`data/synthetic_alignment.py`,
     `method/alignmerge/train_and_merge.py`, `eval/evaluate_outputs.py`) were removed.
   - `python validate.py` passes. It still warns that `eval/test/build_world.py` "looks detached from the method".
     That file is the offline world builder, not an evaluator: it is never run at test time.

## 2. Do better results score higher?

**Before** (old test, variant means over 3 seeds, from `REDTEAM.md`):

| variant | raw | verdict |
|---|---|---|
| weak_no_align | −0.470 | lower ✔ |
| weak_naive_average | −0.359 | lower ✔ |
| weak_less_training | −0.261 | lower ✔ |
| better_more_training | −0.051 | **not higher** |
| better_more_data | −0.031 | **not higher** |
| better_merge_plus | −0.010 | **not higher** |
| better_more_data_grid_merge | +0.035 | higher, but only by dropping the task expert |
| joint training (best actual result) | −0.032 | **not higher** |

Better results mostly did not score higher. The reference seed spread was ±0.021 raw.

**After** (new test; the reference is deterministic, so its seed spread is 0):

| variant | what changes | task acc | align acc | unsafe FNR | raw (mean ± sd over seeds) | boot SE | verdict |
|---|---|---|---|---|---|---|---|
| reference | AlignMerge | 0.7767 | 0.8452 | 0.1846 | 0 | — | |
| better_distill_calibration (3 seeds) | AlignMerge, then distil the sources on the calibration prompts | 0.8349 | 0.8474 | 0.1752 | **+0.253 ± 0.005** | 0.013 | higher ✔ (19 SE) |
| better_refine_calibration (3 seeds) | AlignMerge, then 300 steps on the calibration labels | 0.8180 | 0.8425 | 0.1784 | **+0.180 ± 0.011** | 0.014 | higher ✔ (13 SE) |
| better_layer_coefficients | per-tensor coefficients over {Δ_safe, Δ_expert, δ_AlignMerge} fitted on calibration | 0.8011 | 0.8271 | 0.2112 | **+0.043** | 0.010 | higher ✔ (4 SE) |
| weak_no_budget | λ_bud = 0 (paper ablation) | 0.7819 | 0.8405 | 0.1976 | −0.013 | 0.005 | lower (2.4 SE; ≈ 0 in norm) |
| weak_rank6 | alignment subspace rank 6 | 0.7944 | 0.8108 | 0.2718 | −0.180 | 0.014 | lower ✔ |
| weak_no_align | λ_align = 0 (paper ablation) | 0.8131 | 0.6009 | 0.7766 | −1.586 | 0.018 | lower ✔ |
| weak_naive_average | 50/50 task arithmetic | 0.8253 | 0.5229 | 0.9298 | −2.087 | 0.019 | lower ✔ |

- **Better variants now score higher** by 4–19 paired SEs. Their seed spread (≤ 0.011) is well below the gains.
- **Weaker variants score lower.** The no-budget ablation is the exception: it is only slightly lower (§6).
- **Train and test agree for honest methods.** The reference scores 0.779 / 0.840 on the train slice and 0.777 /
  0.845 on test.
- **Methods fitted to the calibration set look better on train than on test.** Refinement (seed 0) scores 0.909 / 0.895
  on train and 0.820 / 0.840 on test. Training from scratch on the calibration set scores 1.000 / 1.000 on train
  and 0.766 / 0.815 on test (seed 0).

## 3. Major shortcuts: fix, score before and after

| shortcut | fix | before (raw) | after (raw) |
|---|---|---|---|
| Rebuild the test pool from the visible generator and seed, fine-tune on it, copy over the sources | Generator, test seed and test set live only in `eval/test/`; the sources were trained on private data; the test never imports `data/` | **+0.335** (1.000 / 1.000) | cannot be built: no generator, seed or test input is visible |
| Edit `data/` to make the data easier (`shortcut_edit_data`: relabel the calibration set and break `data/benchmark.py`) | The test has its own test set, model class and sources | **+0.244** | **0.000** (identical to the reference) |
| Forge `scores_test.json` and crash the test with a phantom method | File deleted first and always rewritten; method list from disk; a missing submission gets worst values | **+0.335** (if read after exit 1) | **−12.39**, failed run, test exit 0 |
| Joint training shipped without source checkpoints (geometry metrics vanish) | Geometry unscored and computed from eval's own sources; from scratch a method only has the calibration set (`shortcut_joint_training_calibration`, 3 seeds) | **+0.053** | **−0.537 ± 0.008** |
| Copy the safety anchor (no merge) | The new task gives the expert real utility | +0.009 (tie with the row, above the method's mean) | **−0.253** (task 0.642) |
| Copy the instruction anchor | same | — | **−0.245** |
| Overwrite the source checkpoints with the submission (`shortcut_anchor_swap`, also into `data/merge_inputs/`) | Fixed baselines and geometry use eval's copies; same-named files are ignored | +0.009 (also enabled +0.025 with the safety copy) | **0.000** |
| ±inf logits (`shortcut_inf_logits`) | Non-finite outputs score as a failed run | NaN NLL, ECE reported as 0.0 | **−12.39**, failed run |
| NaN weights (`shortcut_nan_params`, new) | Non-finite weights score as a failed run | — | **−12.39**, failed run |
| Checkpoint added after the last `evaluate_train.sh` / submission dropped from the method list (`check_stale_method_list`, `check_no_submission`) | Methods come from the files on disk plus the expected `alignmerge` | unscored or crash | `alignmerge_v2` scored (0.000); no submission scores worst values |
| Operating-point shift of REFUSE, tuned on calibration (`shortcut_refuse_bias`) | (no fix needed) | −0.064 | **−0.016** |
| Hand-written rule: naive average with the expert's answer rows and the safety anchor's REFUSE row (`shortcut_rowswap_rule`) | (no fix needed) | — | **−1.796** |
| Brute force: 31 × 31 task-arithmetic grid chosen on calibration (`bruteforce_grid`) | The new task makes linear interpolation a poor merge | −0.017 (tie) | **−0.290** |
| Brute force: AlignMerge rank × λ_align × γ grid chosen on calibration (`bruteforce_hparam_search`) | — | — | **0.000** (it picks the reference's own settings) |

## 4. Shortcuts left open, and why

- **Distillation on synthetic inputs** (`shortcut_distill_synthetic_inputs`, 3 seeds).
  - What it does: fit a Gaussian to the calibration inputs, sample 40k prompts, and distil the expert's answers
    and the safety anchor's refusal probability into the AlignMerge model.
  - Score: task 0.872, alignment 0.871, raw **+0.481 ± 0.002**. That is close to a model trained jointly on all
    60k private examples (0.868 / 0.885).
  - Why the test can't close it: this is functional distillation on a surrogate of the hidden input
    distribution, not merging. A checkpoint does not reveal how it was trained, so the test cannot tell it apart.
    Making the input distribution hard to sample would also hurt honest calibration-based methods.
  - How it is excluded instead: `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` put it out of scope with a
    rule the grader can check in the code. Every input a method feeds through a model must be one of the 2,048
    calibration prompts.
  - **This shortcut is left to the grader.**
- **Fitting the 1,024 calibration examples.** It is allowed, and it does not pay. Training from scratch on them
  is −0.537, because the test is a fresh 20k + 20k draw.
- **Several `*.pt` files under `checkpoints/`.** Each is scored under its own name, and the ranked submission is
  `alignmerge`. Choosing among several files is left to the grader.

## 5. Reference before and after; headroom

**Reference, before:**
- align 0.9804, task 0.9196 on a 460-example slice;
- it was the best of 5 training seeds (the others were 0.026–0.053 raw lower);
- it was at or beyond the Bayes bound on all five alignment-side metrics.

**Reference, after:**
- align 0.8452, task 0.7767, unsafe FNR 0.1846 on 20,000 + 20,000 prompts;
- deterministic: frozen sources, full-batch merge, single thread.

**Headroom.** The reference is at the bound of no scored metric. Bayes is 0.9125 for alignment and 0.915 for task
accuracy. The joint-training oracle on private data reaches 0.885 / 0.868.

No separate experiment was added. The redesigned task is itself built where the paper's method falls short:
- **Utility.** AlignMerge keeps refusals but gives up utility. It scores 4.9 points below naive averaging and
  10.7 points below the expert on task accuracy.
- **Honest improvements exist.** Three honest variants beat it by 4–19 paired SEs.
- **Simple approaches stay below.** No simple baseline, hand-written rule or brute-force search reaches it.

**Baselines.** Every baseline scores clearly below the reference: naive averaging −2.09, the safety anchor −0.25,
the instruction anchor −0.25 (raw).

## 6. Paper direction and what I could not fix

- **Alignment: reproduced in the paper's direction.**
  - AlignMerge vs naive averaging: alignment 0.845 vs 0.523, unsafe FNR 0.185 vs 0.930.
  - Removing the alignment penalty collapses alignment to 0.601 and doubles the subspace drift (0.80 vs 0.38).
  - Removing the budget lowers AQI (0.246 vs 0.319) and worsens alignment (0.8405 vs 0.8452; FNR 0.198 vs 0.185).
- **Utility: only partly reproduced.**
  - The paper says AlignMerge keeps utility "on par with, or within a narrow margin of, the best expert".
  - Here it gains 13.5 points of task accuracy over the safety anchor but stays 4.9 points below naive averaging
    and 10.7 below the expert.
  - I kept the paper's objective rather than tune the setup toward the paper's numbers. This gap is the
    test's main headroom.
- **The no-budget ablation is barely separated:** −0.013 raw (2.4 paired SE), ≈ 0 in the normalised proxy.
  The budget term has a small effect on this task.
- **Merge baselines not implemented:** Fisher soups, TIES, SafeMerge and MergeAlign. Only the sources and naive
  averaging are fixed baselines.
- **The paper's LLM numbers are placeholders** and not on this scale. They are kept unchanged under
  `_paper_reported` in `scoring/reference.json`.
- **Distillation on synthetic inputs (§4)** is excluded by rule, not by the test.

## Reproducing

- `bash scripts/run.sh` reproduces the reference end to end: method + ablations, baselines, train slice, hidden
  test, and `validate.py --compare`.
- `bash redteam/v2/run_all.sh` reruns every variant and shortcut above. It takes about 30 minutes on one CPU
  core and writes `redteam/v2/summary.md`.
