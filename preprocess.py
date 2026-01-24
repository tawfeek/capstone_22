# preprocess.py
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_LA_PATH = "data/raw/listings LA.csv"
DEFAULT_NYC_PATH = "data/raw/listings NYC.csv"

DEFAULT_TRAIN_CSV = "data/processed/train.csv"
DEFAULT_TEST_CSV = "data/processed/test.csv"
DEFAULT_SUMMARY_JSON = "data/processed/cleaning_summary.json"

TARGET_COL = "review_scores_rating"

# Columns to drop
COLUMNS_TO_DROP = [
    "listing_url", "scrape_id", "source", "picture_url", "host_url",
    "host_thumbnail_url", "host_picture_url", "host_neighbourhood",
    "last_scraped", "first_review", "last_review", "host_since",
    "name", "description", "neighborhood_overview", "host_about",
    "host_verifications", "amenities", "bathrooms_text",
    "host_name",
]


# -----------------------------
# Cleaning helpers
# -----------------------------
def clean_price(s):
    return (
        s.astype(str)
         .str.replace("$", "", regex=False)
         .str.replace(",", "", regex=False)
         .replace("", np.nan)
         .astype(float)
    )


def clean_percentage(s):
    return (
        s.astype(str)
         .str.replace("%", "", regex=False)
         .replace(["", "nan", "N/A"], np.nan)
         .astype(float)
    )


def clean_boolean(s):
    return s.map({"t": 1, "f": 0, True: 1, False: 0}).astype(float)


def extract_amenities_count(s):
    def count(x):
        if pd.isna(x):
            return 0
        try:
            return len(json.loads(x))
        except Exception:
            return str(x).count(",") + 1
    return s.apply(count)


def extract_text_length(s):
    return s.fillna("").astype(str).str.len()


def extract_word_count(s):
    return s.fillna("").astype(str).str.split().str.len()


def extract_host_verifications_count(s):
    def count(x):
        if pd.isna(x):
            return 0
        try:
            return len(json.loads(x.replace("'", '"')))
        except Exception:
            return str(x).count(",") + 1
    return s.apply(count)


# -----------------------------
# Core preprocessing
# -----------------------------
def preprocess(la_path, nyc_path, drop_missing_target=True):
    df_la = pd.read_csv(la_path)
    df_la["city"] = "LA"

    df_nyc = pd.read_csv(nyc_path)
    df_nyc["city"] = "NYC"

    df = pd.concat([df_la, df_nyc], ignore_index=True)

    original_rows = len(df)
    original_cols = len(df.columns)

    if "price" in df:
        df["price"] = clean_price(df["price"])

    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df:
            df[col] = clean_percentage(df[col])

    for col in [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
        "has_availability",
    ]:
        if col in df:
            df[col] = clean_boolean(df[col])

    if "amenities" in df:
        df["amenities_count"] = extract_amenities_count(df["amenities"])

    if "description" in df:
        df["description_length"] = extract_text_length(df["description"])
        df["description_word_count"] = extract_word_count(df["description"])

    if "name" in df:
        df["name_length"] = extract_text_length(df["name"])

    if "neighborhood_overview" in df:
        df["has_neighborhood_overview"] = df["neighborhood_overview"].notna().astype(int)

    if "host_about" in df:
        df["has_host_about"] = df["host_about"].notna().astype(int)
        df["host_about_length"] = extract_text_length(df["host_about"])

    if "host_verifications" in df:
        df["host_verifications_count"] = extract_host_verifications_count(df["host_verifications"])

    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df])

    rows_before_target_drop = len(df)
    if drop_missing_target and TARGET_COL in df:
        df = df.dropna(subset=[TARGET_COL])
    rows_dropped_missing_target = rows_before_target_drop - len(df)

    rows_before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before_dedup - len(df)

    summary = {
        "original_rows": original_rows,
        "original_cols": original_cols,
        "rows_dropped_missing_target": rows_dropped_missing_target,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(df),
        "final_cols": len(df.columns),
    }

    return df, summary


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Preprocess + train/test split")
    parser.add_argument("--la-path", default=DEFAULT_LA_PATH)
    parser.add_argument("--nyc-path", default=DEFAULT_NYC_PATH)
    parser.add_argument("--out-train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out-test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--out-summary", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    df, summary = preprocess(args.la_path, args.nyc_path)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    _ensure_parent(args.out_train_csv)
    _ensure_parent(args.out_test_csv)
    _ensure_parent(args.out_summary)

    train_df.to_csv(args.out_train_csv, index=False)
    test_df.to_csv(args.out_test_csv, index=False)

    summary["train_rows"] = len(train_df)
    summary["test_rows"] = len(test_df)

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("Preprocessing complete")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")


if __name__ == "__main__":
    main()