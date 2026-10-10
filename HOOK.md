# VeriFace — Hook and Automation Specification

## Purpose

This document defines the operational hooks that should govern autonomous Claude Code execution for VeriFace.

`HOOK.md` is the project's human-readable hook policy. If actual Claude Code runtime hooks are configured, they should implement these rules without weakening the research-integrity requirements in `CLAUDE.md`.

---

## 1. Pre-Command Hook

Before executing a non-trivial command:

1. Confirm the current working directory is the VeriFace project root.
2. Check whether the command is read-only, build/test, training, or destructive.
3. Prefer project-local environments.
4. Never execute commands that delete large directories, overwrite datasets, or remove checkpoints unless explicitly required by the current experiment.
5. Record important experiment commands in the experiment log.

---

## 2. Pre-Training Hook

Before every training run:

- verify dataset manifest exists;
- verify class distribution;
- verify train/validation/test separation;
- verify checkpoint directory;
- verify configuration file;
- verify random seed;
- verify device selection;
- run a small batch;
- run one forward/backward pass where training is intended;
- confirm tensor shapes;
- confirm loss is finite;
- confirm output labels are correct.

If the smoke test fails, fix the implementation before launching the full run.

---

## 3. Experiment Hook

Every experiment must create an immutable experiment record containing:

```text
experiment_id
timestamp
git_commit_or_source_hash
model
dataset
split
seed
device
hyperparameters
augmentation_config
training_duration
checkpoint_path
metrics
notes
status
```

Recommended locations:

```text
experiments/<experiment_id>/
├── config.yaml
├── metrics.json
├── training.log
├── curves/
├── confusion_matrix.png
├── roc_curve.png
├── predictions/
└── README.md
```

Never overwrite a completed experiment.

---

## 4. Checkpoint Hook

After successful training:

- save the best validation checkpoint;
- save the final checkpoint;
- save optimizer/scheduler state when relevant;
- save configuration;
- save model architecture metadata;
- calculate a SHA-256 hash for the final checkpoint;
- record the checkpoint in the experiment registry.

If disk space is limited, retain the best checkpoints and experiment metadata first.

---

## 5. Evaluation Hook

Every evaluation must:

1. Load the exact checkpoint used.
2. Verify dataset split.
3. Generate predictions.
4. Calculate metrics.
5. Save raw predictions.
6. Save aggregate metrics.
7. Generate plots where applicable.
8. Record the threshold used.
9. Record whether the result is intra-dataset, cross-dataset or robustness evaluation.

For EER/FAR/FRR experiments, preserve threshold methodology.

---

## 6. Explainability Hook

Before exposing an explanation artifact to the web UI:

- verify it was generated from the selected model;
- verify the artifact dimensions;
- verify the artifact corresponds to the requested media;
- verify the artifact is not an empty/constant heatmap;
- verify overlays align with the original image/frame;
- record the explanation method.

Do not generate fake or random heatmaps to make the interface appear complete.

---

## 7. API Hook

Before considering an API endpoint complete:

- test valid upload;
- test unsupported extension;
- test corrupted file;
- test oversized file;
- test missing file;
- test model failure;
- test timeout/error response;
- test database persistence;
- test report generation.

---

## 8. Frontend Hook

Before considering a UI feature complete:

- verify loading state;
- verify progress state for long video/audio analysis;
- verify success state;
- verify error state;
- verify empty state;
- verify responsive layout;
- verify that confidence and verdict labels are consistent with backend output;
- verify explanation artifacts render correctly.

---

## 9. Report Hook

Whenever a milestone is completed, update:

```text
reports/
├── interim_report/
├── experiments_summary.md
├── model_selection.md
├── reproducibility.md
└── final_project_report.md
```

The interim report must follow the supplied institutional format:

1. Introduction
   - Background
   - Motivation and Scope
   - Problem Statement
   - Proposed Solution
   - Organization of Report
2. Literature Survey
   - Review of current work
   - Identifying gaps
3. Methodology
   - System architecture/Flowcharts
   - Hardware and Software description
   - Algorithm/Data structures
4. Result and Analysis
5. Advantage, Limitations and Applications
   - Advantages
   - Limitations
   - Applications
6. Conclusion and Future Scope
   - Conclusions
   - Future Scope
7. References

Do not invent results for unfinished experiments. Mark them as pending where necessary.

---

## 10. Failure Hook

If a command fails:

1. Capture the error.
2. Determine whether it is:
   - code bug
   - dependency issue
   - device limitation
   - memory limitation
   - dataset issue
   - model incompatibility
   - environment issue
3. Attempt a reasonable fix autonomously.
4. Retry.
5. If the issue remains, create a diagnostic note.
6. Continue with independent tasks where possible.

Do not repeatedly retry the same failing command without changing the hypothesis.

---

## 11. Resource-Awareness Hook

Because the primary local environment is Apple Silicon/M4:

- detect available MPS;
- use MPS when supported;
- use CPU fallback when required;
- reduce batch size if memory is insufficient;
- use reduced smoke-test datasets before full training;
- use mixed precision only when verified stable on the active device;
- avoid unnecessarily large concurrent jobs.

Never sacrifice reproducibility merely to make training faster.

---

## 12. Git Hook Policy

Before major milestones:

- check `git status`;
- create a meaningful commit;
- never commit datasets, secrets, API keys, passwords, or massive checkpoints unless explicitly intended;
- maintain `.gitignore`;
- record experiment IDs in commit messages where useful.

---

## 13. Final Verification Hook

Before reporting the project as complete, execute:

```text
1. Environment validation
2. Dataset validation
3. Model smoke tests
4. Best-checkpoint loading
5. Image inference
6. Video inference
7. Audio inference
8. Explanation generation
9. API tests
10. Frontend build
11. Database verification
12. End-to-end workflow
13. Forensic PDF generation
14. Report update
15. Reproducibility check
```

Only then mark the project as production-ready/demo-ready.

