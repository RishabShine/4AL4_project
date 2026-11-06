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
tuning, not independent evidence of final performance. Test data remains reserved.
