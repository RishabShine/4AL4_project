"""Fit regression candidates using only the training split.

Run from the project root with: python3 scripts/train.py --run v1
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "preprocessed" / "train.csv"

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


def train_candidates(X_train, y_train, alphas):
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

    # checks how much the features add beyond a constant training average prediction.
    mean_model = Pipeline([
        ("scale", StandardScaler()),
        ("regression", DummyRegressor(strategy="mean")),
    ])
    mean_model.fit(X_train, y_train)
    models["training_mean_baseline"] = mean_model
    rows.append({
        "model": "training_mean_baseline",
        "alpha": None,
        "split": "train",
        **evaluate(y_train, mean_model.predict(X_train)),
    })

    # train ordinary regression once, then a separate Ridge model for each alpha.
    candidates = [("linear_regression", None)] + [
        (f"ridge_alpha_{alpha:g}", alpha) for alpha in alphas
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v1", help="Name for this experiment, e.g. v2")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    args = parser.parse_args()
    if not args.run or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.run):
        parser.error("Run names can contain only letters, numbers, underscores, and hyphens.")
    if not all(np.isfinite(alpha) and alpha > 0 for alpha in args.alphas):
        parser.error("Alphas must be positive finite numbers.")
    alphas = list(dict.fromkeys(args.alphas))
    model_file = ROOT / "models" / args.run / "training_candidates.joblib"
    results_dir = ROOT / "results" / "train" / args.run
    if model_file.exists() or results_dir.exists():
        parser.error("This training run already exists. Use a new --run name to preserve it.")

    X_train, y_train = load_training_data()
    models, results = train_candidates(X_train, y_train, alphas)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run": args.run,
        "alphas": alphas,
        "training_file_sha256": hashlib.sha256(TRAIN_FILE.read_bytes()).hexdigest(),
        "training_examples": len(X_train),
        "feature_columns": X_train.columns.tolist(),
        "target_column": TARGET_COLUMN,
    }
    joblib.dump({
        "models": models,
        **metadata,
    }, model_file)
    results.to_csv(results_dir / "metrics.csv", index=False)
    with (results_dir / "config.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Trained {len(models)} candidates on {len(X_train)} examples with {X_train.shape[1]} features.")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved models/{args.run}/training_candidates.joblib and results/train/{args.run}/.")
    print("These are training errors only; model selection comes next on validation data.")


if __name__ == "__main__":
    main()
