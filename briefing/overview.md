# Overview

AlignMerge addresses a failure of weight-space model merging: it keeps utility but erodes alignment. A merge that
pulls in a task expert's update also pulls in the expert's forgotten refusals. AlignMerge adds Fisher-geometry
and alignment-budget terms to the merge objective. The merged model stays close to the useful experts in the task
geometry without moving along alignment-sensitive directions.

The gym reproduces this on a small synthetic model family. Three frozen source checkpoints (instruction anchor,
safety anchor, task expert) and a small labelled calibration set are given. A merged checkpoint is scored on a
hidden test set for both refusal behaviour on unsafe prompts and task utility.
