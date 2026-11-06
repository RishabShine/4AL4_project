"""Compare the trained candidates on later seasons without refitting them.

Run after train.py with: python3 scripts/validate.py --run v1
"""

import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from train import IDENTIFIER_COLUMNS, ROOT, TRAIN_FILE, evaluate


VALIDATION_FILE = ROOT / "data" / "preprocessed" / "validation.csv"


def load_validation_data(bundle):
    data = pd.read_csv(VALIDATION_FILE)

    # using the same feature columns and order that were used during training.
    X_validation = data[bundle["feature_columns"]]
    y_validation = data[bundle["target_column"]]
    if data.empty:
        raise ValueError("Validation data contains no examples.")
    if not np.isfinite(X_validation.to_numpy(dtype=float)).all():
        raise ValueError("Validation features contain missing or infinite values.")
    if not np.isfinite(y_validation.to_numpy(dtype=float)).all() or not y_validation.between(0, 1).all():
        raise ValueError("Validation targets must be finite and between 0 and 1.")
    if data.duplicated(["team_id", "target_season"]).any():
        raise ValueError("Validation data contains duplicate team-season examples.")

    # check the time boundary, using identifiers only, not training outcomes.
    train_years = pd.read_csv(ROOT / "data" / "preprocessed" / "train.csv", usecols=["target_season"])
    if data["target_season"].min() <= train_years["target_season"].max():
        raise ValueError("Validation target seasons must come after training seasons.")

    predictions = data[IDENTIFIER_COLUMNS + [bundle["target_column"]]].copy()
    return X_validation, y_validation, predictions


def validate_candidates(models, X_validation, y_validation, predictions):
    rows = []
    predictions["previous_season_baseline"] = X_validation["win_pct_lag1"]
    rows.append({
        "model": "previous_season_baseline",
        "alpha": None,
        "split": "validation",
        **evaluate(y_validation, predictions["previous_season_baseline"]),
    })

    for name, model in models.items():
        y_pred = model.predict(X_validation)
        predictions[name] = y_pred
        rows.append({
            "model": name,
            "alpha": model.named_steps["regression"].get_params().get("alpha"),
            "split": "validation",
            **evaluate(y_validation, y_pred),
        })

    # looking for lower MSE
    results = pd.DataFrame(rows).sort_values("mse", kind="stable").reset_index(drop=True)
    return results, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v1", help="Which saved training run to evaluate")
    args = parser.parse_args()
    if not args.run or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.run):
        parser.error("Run names can contain only letters, numbers, underscores, and hyphens.")
    model_file = ROOT / "models" / args.run / "training_candidates.joblib"
    results_dir = ROOT / "results" / "validation" / args.run
    bundle = joblib.load(model_file)
    if "training_file_sha256" in bundle and bundle["training_file_sha256"] != hashlib.sha256(TRAIN_FILE.read_bytes()).hexdigest():
        raise ValueError("Training data changed since this run was fitted. Create a new training run.")
    X_validation, y_validation, predictions = load_validation_data(bundle)
    results, predictions = validate_candidates(
        bundle["models"], X_validation, y_validation, predictions
    )

    # also including the baseline in the comparison rather than assume a learned model wins
    best = results.iloc[0]
    selection = {
        "run": args.run,
        "model": best["model"],
        "alpha": None if pd.isna(best["alpha"]) else float(best["alpha"]),
        "selection_metric": "validation_mse",
        "validation_mse": float(best["mse"]),
        "validation_mae": float(best["mae"]),
        "validation_examples": len(y_validation),
        "validation_file_sha256": hashlib.sha256(VALIDATION_FILE.read_bytes()).hexdigest(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "metrics.csv", index=False)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    with (results_dir / "selection.json").open("w") as file:
        json.dump(selection, file, indent=2)
        file.write("\n")

    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Selected by validation MSE: {best['model']} ({len(y_validation)} examples).")
    print(f"Saved results/validation/{args.run}/. Test data remains unused.")


if __name__ == "__main__":
    main()
