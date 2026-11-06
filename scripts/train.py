"""Fit regression candidates using only the training split.

Run from the project root with: python3 scripts/train.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "preprocessed" / "train.csv"
MODEL_FILE = ROOT / "models" / "training_candidates.joblib"
RESULTS_FILE = ROOT / "results" / "training_metrics.csv"

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
IDENTIFIER_COLUMNS = ["team_id", "team", "target_season"]
TARGET_COLUMN = "target_win_pct"


def load_training_data():
    data = pd.read_csv(TRAIN_FILE)

    # team names and years identify the rows, the label must not become an input.
    X_train = data.drop(columns=IDENTIFIER_COLUMNS + [TARGET_COLUMN])
    y_train = data[TARGET_COLUMN]

    if data.empty or X_train.shape[1] == 0:
        raise ValueError("Training data must contain examples and features.")
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain missing or infinite values.")
    if not np.isfinite(y_train.to_numpy(dtype=float)).all() or not y_train.between(0, 1).all():
        raise ValueError("Target win percentages must be finite and between 0 and 1.")
    if data.duplicated(["team_id", "target_season"]).any():
        raise ValueError("Training data contains duplicate team season examples.")

    return X_train, y_train


def training(X_train, y_train, alpha=None):
    regression = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("regression", regression),
    ])

    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred):
    # prediction errors, not adding the Ridge weight penalty.
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def train_candidates(X_train, y_train):
    models = {}
    rows = []

    # the baseline just repeats last season's win percentage
    baseline_predictions = X_train["win_pct_lag1"].to_numpy()
    rows.append({
        "model": "previous_season_baseline",
        "alpha": None,
        "split": "train",
        **evaluate(y_train, baseline_predictions),
    })

    # train ordinary regression once, then a separate Ridge model for each alpha.
    candidates = [("linear_regression", None)] + [
        (f"ridge_alpha_{alpha:g}", alpha) for alpha in ALPHAS
    ]
    for name, alpha in candidates:
        model = training(X_train, y_train, alpha)
        models[name] = model
        predictions = model.predict(X_train)
        rows.append({
            "model": name,
            "alpha": alpha,
            "split": "train",
            **evaluate(y_train, predictions),
        })

    # validation will compare them without refitting.
    return models, pd.DataFrame(rows)


def main():
    X_train, y_train = load_training_data()
    models, results = train_candidates(X_train, y_train)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models,
        "feature_columns": X_train.columns.tolist(),
        "alphas": ALPHAS,
        "target_column": TARGET_COLUMN,
    }, MODEL_FILE)
    results.to_csv(RESULTS_FILE, index=False)

    print(f"Trained {len(models)} candidates on {len(X_train)} examples with {X_train.shape[1]} features.")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Saved models/training_candidates.joblib and results/training_metrics.csv.")
    print("These are training errors only; model selection comes next on validation data.")


if __name__ == "__main__":
    main()
