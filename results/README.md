# Experiment results

Use the same scripts for each experiment, with a separate run name and alpha list.
This keeps the learning/evaluation code consistent when comparing settings.

```text
results/
  train/
    v1/metrics.csv
    v2/metrics.csv
    v2/config.json
  validation/
    v1/metrics.csv
    v1/predictions.csv
    v1/selection.json
    v2/metrics.csv
    v2/predictions.csv
    v2/selection.json
  validation_comparison.csv
models/
  v1/training_candidates.joblib
  v2/training_candidates.joblib
```

The original v1 files were moved into this layout unchanged. The original run
used alphas `[0.01, 0.1, 1, 10, 100, 1000]`. Its selection JSON predates run
metadata and its training folder has no config file. New training runs save
settings, feature order, and a training-file checksum in `config.json`.
Saved model binaries are ignored by Git and can be regenerated through training.

## Extended alpha experiment

```bash
.venv/bin/python scripts/train.py --run v2 --alphas 0.01 0.1 1 10 100 300 1000 3000 10000 100000
.venv/bin/python scripts/validate.py --run v2
```

Training refuses to overwrite an existing run. Use a new name for a new
experiment. Validation evaluates the named run without fitting or changing its
models; rerunning it replaces only that run's validation outputs. Commands resolve
paths from the repository location rather than the current working directory.

Both experiments use the same 23 features and chronological splits. New runs also
include a training-mean baseline: a constant prediction learned only from training
labels. This helps interpret the effect of very strong Ridge regularization.
The original v1 outputs are preserved without adding that baseline retroactively.

MSE is the selection metric; MAE is also reported. The comparison CSV summarizes
these two runs and their previous-season baseline. This is validation-driven
tuning, not independent evidence of final performance. Test data was reserved
through both tuning runs, then evaluated separately as described below.

## Test evaluation

```bash
.venv/bin/python scripts/test.py --run v2
```

The test script loads the validation-selected model from the saved run and calls
`predict` without refitting its weights or scaler. It evaluates only that choice
and the baselines, not every alpha. All fitted values still come exclusively from
the 390 training examples. It saves `metrics.csv`, `predictions.csv`, and
`evaluation.json` under `results/test/v2/`. Rerunning replaces those outputs.

The test contains 90 team-season examples, with target ending years 2024-2026.
Metrics include MSE, MAE, RMSE (square root of MSE), and RSS (sum of squared
prediction errors, equal to MSE times the number of examples). None includes
the Ridge weight penalty. MSE and MAE are the main report metrics; RSS depends
on sample count. The selected model and alpha are fixed before test evaluation.
The observed test results should not guide further tuning of this experiment.
