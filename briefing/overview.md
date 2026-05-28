# Overview

AlignMerge addresses the failure mode where weight-space model merging preserves utility but erodes alignment. It adds explicit Fisher-geometry and AQI-budget constraints to the merge so the result remains close to useful experts without moving far along safety-sensitive directions.

The reproduction package implements the method on a small synthetic multitask neural model, with reusable scripts for running the merge and evaluating the resulting artifacts.

