"""Evaluate the validation-selected model without changing its weights.

Run with: python3 scripts/test.py --run v2
"""

import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from train import IDENTIFIER_COLUMNS, ROOT, TRAIN_FILE, evaluate


TEST_FILE = ROOT / "data" / "preprocessed" / "test.csv"
VALIDATION_FILE = ROOT / "data" / "preprocessed" / "validation.csv"


def load_test_data(bundle):
    data = pd.read_csv(TEST_FILE)
    X_test = data[bundle["feature_columns"]]
    y_test = data[bundle["target_column"]]
    if data.empty or not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Test features must contain examples with finite values.")
    if not np.isfinite(y_test.to_numpy(dtype=float)).all() or not y_test.between(0, 1).all():
        raise ValueError("Test targets must be finite and between 0 and 1.")
    if data.duplicated(["team_id", "target_season"]).any():
        raise ValueError("Test data contains duplicate team-season examples.")
    validation_years = pd.read_csv(VALIDATION_FILE, usecols=["target_season"])
    if data["target_season"].min() <= validation_years["target_season"].max():
        raise ValueError("Test target seasons must come after validation seasons.")

    predictions = data[IDENTIFIER_COLUMNS + [bundle["target_column"]]].copy()
    return X_test, y_test, predictions


def test_selected_model(bundle, selected_name, X_test, y_test, predictions):
    predictions["previous_season_baseline"] = X_test["win_pct_lag1"]
    names = ["previous_season_baseline"]
    if "training_mean_baseline" in bundle["models"]:
        names.append("training_mean_baseline")
    if selected_name not in names:
        names.append(selected_name)

    rows = []
    for name in names:
        alpha = None
        if name != "previous_season_baseline":
            model = bundle["models"][name]
            predictions[name] = model.predict(X_test)
            alpha = model.named_steps["regression"].get_params().get("alpha")
        metrics = evaluate(y_test, predictions[name])
        errors = y_test.to_numpy() - predictions[name].to_numpy()
        rows.append({
            "model": name,
            "alpha": alpha,
            "split": "test",
            "n_examples": len(y_test),
            **metrics,
            "rmse": np.sqrt(metrics["mse"]),
            "rss": np.sum(errors ** 2),
        })
    return pd.DataFrame(rows), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v2", help="Run whose validation choice will be tested")
    args = parser.parse_args()
    if not args.run or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.run):
        parser.error("Run names can contain only letters, numbers, underscores, and hyphens.")

    model_file = ROOT / "models" / args.run / "training_candidates.joblib"
    selection_file = ROOT / "results" / "validation" / args.run / "selection.json"
    bundle = joblib.load(model_file)
    selection = json.loads(selection_file.read_text())
    for key, path, metadata in [
        ("training_file_sha256", TRAIN_FILE, bundle),
        ("validation_file_sha256", VALIDATION_FILE, selection),
    ]:
        if key in metadata and metadata[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{path.name} changed since training/selection.")
    selected_name = selection["model"]
    if selected_name != "previous_season_baseline" and selected_name not in bundle["models"]:
        raise ValueError("The selected model is missing from the saved training run.")

    X_test, y_test, predictions = load_test_data(bundle)
    results, predictions = test_selected_model(
        bundle, selected_name, X_test, y_test, predictions
    )
    results_dir = ROOT / "results" / "test" / args.run
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "metrics.csv", index=False)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    with (results_dir / "evaluation.json").open("w") as file:
        json.dump({
            "run": args.run,
            "selected_model": selected_name,
            "alpha": selection["alpha"],
            "refitted": False,
            "test_examples": len(y_test),
            "test_target_years": sorted(predictions["target_season"].unique().tolist()),
            "model_file_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            "test_file_sha256": hashlib.sha256(TEST_FILE.read_bytes()).hexdigest(),
        }, file, indent=2)
        file.write("\n")

    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved results/test/{args.run}/. Used {selected_name} without refitting.")


if __name__ == "__main__":
    main()
