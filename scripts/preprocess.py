
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Team Summaries.csv"
OUTPUT_DIR = ROOT / "data" / "preprocessed"

# Kaggle labels seasons by ending year: 2005 means 2004-05.
FIRST_HISTORY_SEASON = 2005
LAST_COMPLETED_SEASON = 2026
HISTORY_LENGTH = 3
SPLITS = {
    "train": (2008, 2020),
    "validation": (2021, 2023),
    "test": (2024, 2026),
}

# Follow the same basketball organization through modern moves/name changes.
# These mappings apply to our 2005-onward period, not all historical NBA records.
TEAM_ALIASES = {
    "NJN": "BRK",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHA": "CHO",
}

HISTORICAL_STATS = {
    "win_pct": "win_pct",
    "o_rtg": "off_rating",
    "d_rtg": "def_rating",
    "e_fg_percent": "efg_pct",
    "tov_percent": "tov_pct",
    "orb_percent": "orb_pct",
    "drb_percent": "drb_pct",
}


def load_team_seasons():
    raw = pd.read_csv(RAW_FILE)
    required = {
        "season", "lg", "team", "abbreviation", "w", "l", "n_rtg",
        *[column for column in HISTORICAL_STATS if column != "win_pct"],
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    teams = raw.loc[
        raw["lg"].eq("NBA")
        & raw["abbreviation"].notna()
        & raw["team"].ne("League Average")
        & raw["season"].between(FIRST_HISTORY_SEASON, LAST_COMPLETED_SEASON)
    ].copy()
    numeric_columns = ["w", "l", "n_rtg"] + [
        column for column in HISTORICAL_STATS if column != "win_pct"
    ]
    teams[numeric_columns] = teams[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    records = teams[["w", "l"]]
    if (
        records.isna().any().any()
        or not np.isfinite(records).all().all()
        or records.lt(0).any().any()
        or records.mod(1).ne(0).any().any()
        or records.sum(axis=1).le(0).any()
    ):
        raise ValueError("Each team-season must have valid wins and losses.")

    teams["team_id"] = teams["abbreviation"].replace(TEAM_ALIASES)
    if teams.duplicated(["team_id", "season"]).any():
        raise ValueError("Duplicate team-season records after team mapping.")

    # Use actual games played, including shortened seasons, rather than 82.
    teams["win_pct"] = teams["w"] / (teams["w"] + teams["l"])
    teams = teams.sort_values(["team_id", "season"]).reset_index(drop=True)
    print(f"Source: {len(raw)} rows; retained {len(teams)} NBA team-seasons.")
    return teams


def build_examples(teams):
    examples = teams[["team_id", "team", "season"]].rename(
        columns={"season": "target_season"}
    ).copy()
    grouped = teams.groupby("team_id", sort=False)
    feature_columns = []
    valid_history = pd.Series(True, index=teams.index)

    for lag in range(1, HISTORY_LENGTH + 1):
        # A shift alone could jump a missing season; require the exact year.
        historical_season = grouped["season"].shift(lag)
        valid_history &= historical_season.eq(teams["season"] - lag)
        for source, name in HISTORICAL_STATS.items():
            column = f"{name}_lag{lag}"
            examples[column] = grouped[source].shift(lag)
            feature_columns.append(column)

    # Both changes use completed seasons t-1 and t-2, never target season t.
    examples["win_pct_change"] = (
        examples["win_pct_lag1"] - examples["win_pct_lag2"]
    )
    examples["net_rating_change"] = (
        grouped["n_rtg"].shift(1) - grouped["n_rtg"].shift(2)
    )
    feature_columns += ["win_pct_change", "net_rating_change"]
    examples["target_win_pct"] = teams["win_pct"]

    numeric = examples[feature_columns + ["target_win_pct"]]
    valid_values = pd.Series(
        np.isfinite(numeric.to_numpy()).all(axis=1), index=teams.index
    )
    print(f"Excluded {int((~valid_history).sum())} rows without three consecutive prior seasons.")
    print(f"Excluded {int((valid_history & ~valid_values).sum())} additional rows with missing/nonfinite values.")
    return examples.loc[valid_history & valid_values].sort_values(
        ["target_season", "team_id"]
    ).reset_index(drop=True), feature_columns


def main():
    teams = load_team_seasons()
    examples, features = build_examples(teams)
    partitions = {}
    for name, (first, last) in SPLITS.items():
        partition = examples.loc[examples["target_season"].between(first, last)]
        if partition.empty:
            raise ValueError(f"No examples in {name} split.")
        partitions[name] = partition

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, partition in partitions.items():
        path = OUTPUT_DIR / f"{name}.csv"
        partition.to_csv(path, index=False)
        print(f"{name}: {len(partition)} examples, target years {SPLITS[name]}, saved {path.relative_to(ROOT)}")
    print(f"Each example has {len(features)} features, 3 identifiers, and target_win_pct.")
    print("Scaling, model fitting, and model selection happen later.")


if __name__ == "__main__":
    main()