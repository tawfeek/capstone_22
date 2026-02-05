# preprocess.py
import argparse
import json
import os
import re
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

from utils import _ensure_parent_dir

# # -----------------------------
# # Default paths (edit as needed)
# # -----------------------------
# DEFAULT_LA_PATH = "data/raw/listingsLA.csv"
# DEFAULT_NYC_PATH = "data/raw/listingsNYC.csv"
#
# DEFAULT_OUTPUT_CSV = "data/processed/listings_combined_clean.csv"
# DEFAULT_SUMMARY_JSON = "data/processed/cleaning_summary.json"
#
# TARGET_COL = "review_scores_rating"

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_OUTPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_SUMMARY_JSON = "data/processed/cleaning_summary.json"

TARGET_COL = "review_scores_rating"

# Columns to drop after feature extraction (IDs, URLs, raw text, dates)
COLUMNS_TO_DROP = [
    # IDs and URLs - don't generalize
    "listing_url", "scrape_id", "source", "picture_url", "host_url",
    "host_thumbnail_url", "host_picture_url", "host_neighbourhood",
    # Dates - will be used for feature extraction then dropped
    "last_scraped", "first_review", "last_review", "host_since",
    # Raw text - will be used for feature extraction then dropped
    "name", "description", "neighborhood_overview", "host_about",
    "host_verifications", "amenities", "bathrooms_text",
    # Names - PII, don't generalize
    "host_name",
]

TARGET_ENCODE_COLS = ["neighbourhood_cleansed", "property_type"]

# Columns to exclude from final features
# See md_files/FEATURE_EXCLUSIONS.md for detailed reasoning
COLS_TO_EXCLUDE = [
    # "city",                  # Generalization - model should work on any city
    "host_id",  # High cardinality identifier - causes overfitting
    "room_type",  # Redundant with is_entire_home, is_private_room binary flags
    # Note: neighbourhood_cleansed and property_type are NOW handled by TargetEncoder in pipeline

    # Availability Bloat - 8 variables for "minimum nights" is extreme collinearity
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90",  # Subsets of availability_365

    # Review Redundancy - correlate heavily with reviews_per_month
    "number_of_reviews_ltm", "number_of_reviews_l30d",

    # Host Count Redundancy - breakdown of host_listings_count
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "host_total_listings_count",  # Duplicate of host_listings_count

    # Low Variance - nearly always 1 (True), no splitting power
    "host_has_profile_pic",

    # Negative Permutation Importance - adding noise, not signal
    # See results/diagnostics/permutation_importance.csv
    "has_washer", "description_word_count", "has_kitchen", "is_entire_home",
    "has_dryer", "has_self_checkin", "has_pets_allowed", "has_free_parking",
    "is_private_room", "has_neighborhood_overview", "has_carbon_alarm",
    "host_about_length", "has_tv", "bedrooms", "host_acceptance_rate",
    "bathrooms", "has_fire_extinguisher", "name_length", "has_pool",
    "accommodates", "has_ac", "beds",

    # Replaced by log transforms (avoid collinearity with log_ versions)
    "number_of_reviews",  # Use log_number_of_reviews instead
    "host_listings_count",  # Use log_host_listings_count instead (dropped in preprocess)
]



def clean_price(price_series: pd.Series) -> pd.Series:
    """Convert price from '$1,234.56' format to float."""
    return (
        price_series
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def clean_percentage(pct_series: pd.Series) -> pd.Series:
    """Convert percentage from '95%' format to float (0-100)."""
    return (
        pct_series
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["", "nan", "N/A"], np.nan)
        .astype(float)
    )


def clean_boolean(bool_series: pd.Series) -> pd.Series:
    """Convert 't'/'f' to 1/0."""
    mapping = {"t": 1, "f": 0, True: 1, False: 0}
    return bool_series.map(mapping).astype(float)


def extract_amenities_count(amenities_series: pd.Series) -> pd.Series:
    """Count number of amenities from JSON-like string."""

    def count_amenities(x):
        if pd.isna(x):
            return 0
        try:
            # Amenities are stored as JSON array string: '["Wifi", "Kitchen", ...]'
            items = json.loads(x)
            return len(items)
        except (json.JSONDecodeError, TypeError):
            # Fallback: count commas
            return str(x).count(",") + 1 if str(x).strip() else 0

    return amenities_series.apply(count_amenities)


def parse_amenities(amenities_str: str) -> set:
    """Parse amenities JSON string into a set of lowercase amenity names."""
    if pd.isna(amenities_str):
        return set()
    try:
        items = json.loads(amenities_str)
        return set(item.lower() for item in items)
    except (json.JSONDecodeError, TypeError):
        return set()


def extract_key_amenities(amenities_series: pd.Series) -> pd.DataFrame:
    """
    Extract binary features for key amenities that impact guest satisfaction.
    Returns a DataFrame with binary columns for each key amenity.
    """
    # Define key amenities to extract (grouped by category)
    key_amenities = {
        # Essential amenities
        "has_wifi": ["wifi", "wireless internet"],
        "has_kitchen": ["kitchen"],
        "has_ac": ["air conditioning", "central air conditioning"],
        "has_heating": ["heating", "central heating"],
        "has_washer": ["washer", "washer / dryer"],
        "has_dryer": ["dryer", "washer / dryer"],
        # Parking
        "has_free_parking": ["free parking on premises", "free street parking", "free parking"],
        # Work-friendly
        "has_workspace": ["dedicated workspace"],
        # Comfort
        "has_tv": ["tv", "hdtv"],
        "has_hot_water": ["hot water"],
        # Self-service
        "has_self_checkin": ["self check-in", "lockbox", "keypad", "smart lock"],
        # Safety
        "has_smoke_alarm": ["smoke alarm", "smoke detector"],
        "has_carbon_alarm": ["carbon monoxide alarm", "carbon monoxide detector"],
        "has_fire_extinguisher": ["fire extinguisher"],
        "has_first_aid": ["first aid kit"],
        # Luxury/premium
        "has_pool": ["pool", "private pool", "shared pool"],
        "has_hot_tub": ["hot tub", "jacuzzi"],
        "has_gym": ["gym", "fitness center"],
        # Pet-friendly
        "has_pets_allowed": ["pets allowed"],
    }

    # Parse all amenities once
    parsed = amenities_series.apply(parse_amenities)

    # Create binary columns
    result = pd.DataFrame(index=amenities_series.index)
    for feature_name, keywords in key_amenities.items():
        result[feature_name] = parsed.apply(
            lambda amenities: int(any(kw in amenities for kw in keywords))
        )

    return result


def extract_host_experience_days(host_since_series: pd.Series, reference_date: str = None) -> pd.Series:
    """
    Calculate days since host joined Airbnb.
    Uses the max date in the series as reference if not provided.
    """
    dates = pd.to_datetime(host_since_series, errors="coerce")
    if reference_date is None:
        reference = dates.max()
    else:
        reference = pd.to_datetime(reference_date)

    days = (reference - dates).dt.days
    return days.fillna(0).astype(int)


def encode_response_time(response_time_series: pd.Series) -> pd.Series:
    """
    Encode host_response_time as numeric (lower is better).
    """
    encoding = {
        "within an hour": 1,
        "within a few hours": 2,
        "within a day": 3,
        "a few days or more": 4,
    }
    return response_time_series.map(encoding).fillna(5)  # 5 = unknown/no response


def extract_text_length(text_series: pd.Series) -> pd.Series:
    """Extract character length from text column."""
    return text_series.fillna("").astype(str).str.len()


def extract_word_count(text_series: pd.Series) -> pd.Series:
    """Extract word count from text column."""
    return text_series.fillna("").astype(str).str.split().str.len()


def extract_host_verifications_count(verif_series: pd.Series) -> pd.Series:
    """Count number of host verifications."""

    def count_verifications(x):
        if pd.isna(x):
            return 0
        try:
            items = json.loads(x.replace("'", '"'))
            return len(items)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return str(x).count(",") + 1 if str(x).strip() else 0

    return verif_series.apply(count_verifications)


def extract_luxury_score(text_series: pd.Series) -> pd.Series:
    """
    Count premium/positive keywords in text.
    Words derived from analysis of 4.9+ star reviews.
    """
    luxury_words = [
        "view", "ocean", "luxury", "renovated", "heart", "spacious",
        "private", "quiet", "spotless", "pristine", "modern", "boutique"
    ]
    text = text_series.fillna("").str.lower()
    return text.apply(lambda x: sum(word in x for word in luxury_words))


def extract_warning_score(text_series: pd.Series) -> pd.Series:
    """
    Count negative/basic keywords in text.
    Words derived from analysis of <4.5 star reviews.
    """
    warning_words = [
        "old", "noisy", "loud", "basic", "simple", "party", "student",
        "dated", "worn", "smell", "street", "thin walls", "shared"
    ]
    text = text_series.fillna("").str.lower()
    return text.apply(lambda x: sum(word in x for word in warning_words))


def apply_log_transforms(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply log1p transform to skewed features.
    Creates new log_* columns and drops the originals to avoid collinearity.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # Create log-transformed column
            df[f"log_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))
            # Drop original to avoid collinearity
            df = df.drop(columns=[col])
        else:
            df[f"log_{col}"] = None
            # df = df.drop(columns=[col])
    return df


def run_initial_cleaning(dfs, drop_missing_target: bool = True,is_training:bool = False):
    """
    Loads LA + NYC listings, performs feature engineering and cleaning.

    Steps:
    1. Load and combine datasets
    2. Clean price, percentage, and boolean columns
    3. Extract features from text columns (before dropping)
    4. Drop unnecessary columns
    5. Drop rows with missing target (optional)
    6. Remove duplicates

    Returns (df_clean, cleaning_summary).
    """
    # if isinstance(file_inputs, str):
    #     paths = [file_inputs]
    # elif isinstance(file_inputs, list):
    #     paths = file_inputs
    # else:
        # raise ValueError("file_inputs must be a string path or a list of paths.")

    processed_dfs = []
    total_original_rows = 0
    total_original_cols = 0
    for df in dfs:
        # if not os.path.exists(path):
        #     print(f"Warning: Path {path} does not exist. Skipping.")
        #     continue

        # df = pd.read_csv(path)
        total_original_rows += len(df)
        total_original_cols = max(total_original_cols, len(df.columns))
        # =========================================
        # 1. CLEAN PRICE COLUMN
        # =========================================
        if "price" in df.columns:
            df["price"] = clean_price(df["price"])

        # =========================================
        # 2. CLEAN PERCENTAGE COLUMNS
        # =========================================
        pct_cols = ["host_response_rate", "host_acceptance_rate"]
        for col in pct_cols:
            if col in df.columns:
                df[col] = clean_percentage(df[col])

        # =========================================
        # 3. CLEAN BOOLEAN COLUMNS
        # =========================================
        bool_cols = ["host_is_superhost", "host_has_profile_pic",
                     "host_identity_verified", "instant_bookable", "has_availability"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = clean_boolean(df[col])

        # =========================================
        # 4. FEATURE EXTRACTION FROM TEXT COLUMNS
        # =========================================
        # Extract features BEFORE dropping text columns

        # Amenities count + key amenity binary features
        if "amenities" in df.columns:
            df["amenities_count"] = extract_amenities_count(df["amenities"])
            # Extract key amenities as binary features
            key_amenities_df = extract_key_amenities(df["amenities"])
            df = pd.concat([df, key_amenities_df], axis=1)
        if "estimated_occupancy_l365d" not in df.columns:
            df["estimated_occupancy_l365d"] = None
        # Description features
        if "description" in df.columns:
            df["description_length"] = extract_text_length(df["description"])
            df["description_word_count"] = extract_word_count(df["description"])
            # Text mining: luxury and warning keywords
            df["luxury_count"] = extract_luxury_score(df["description"])
            df["warning_count"] = extract_warning_score(df["description"])
        else:
            df["description_length"] = None
            df["description_word_count"] = None
            df["luxury_count"] = None
            df["warning_count"] = None
        # Name length
        if "name" in df.columns:
            df["name_length"] = extract_text_length(df["name"])

        # Neighborhood overview (binary: has or not)
        if "neighborhood_overview" in df.columns:
            df["has_neighborhood_overview"] = df["neighborhood_overview"].notna().astype(int)

        # Host about (binary + length)
        if "host_about" in df.columns:
            df["has_host_about"] = df["host_about"].notna().astype(int)
            df["host_about_length"] = extract_text_length(df["host_about"])
        else:
            df["has_host_about"] = None
            df["host_about_length"] = None
        # Host verifications count
        if "host_verifications" in df.columns:
            df["host_verifications_count"] = extract_host_verifications_count(df["host_verifications"])
        else:
            df["host_verifications_count"] = None
        # =========================================
        # 4b. HOST EXPERIENCE (days since joining)
        # =========================================
        if "host_since" in df.columns:
            df["host_experience_days"] = extract_host_experience_days(df["host_since"])

        # =========================================
        # 4c. RESPONSE TIME ENCODING (numeric)
        # =========================================
        if "host_response_time" in df.columns:
            df["response_time_score"] = encode_response_time(df["host_response_time"])

        # =========================================
        # 4d. RATIO FEATURES (value/comfort metrics)
        # =========================================
        # Price per person (value for money)
        if "price" in df.columns and "accommodates" in df.columns:
            df["price_per_person"] = df["price"] / df["accommodates"].replace(0, 1)

        # Space comfort ratios
        if "bedrooms" in df.columns and "accommodates" in df.columns:
            df["bedrooms_per_person"] = df["bedrooms"] / df["accommodates"].replace(0, 1)

        if "beds" in df.columns and "accommodates" in df.columns:
            df["beds_per_person"] = df["beds"] / df["accommodates"].replace(0, 1)

        # Bathroom ratio
        if "bathrooms" in df.columns and "accommodates" in df.columns:
            df["bathrooms_per_person"] = df["bathrooms"] / df["accommodates"].replace(0, 1)

        # Interaction: Minimum stay commitment cost
        # High price + long minimum stay = higher risk for guests
        if "price" in df.columns and "minimum_nights" in df.columns:
            df["min_stay_cost"] = df["price"] * df["minimum_nights"]

        # =========================================
        # 4e. ROOM TYPE ENCODING
        # =========================================
        if "room_type" in df.columns:
            df["is_entire_home"] = (df["room_type"] == "Entire home/apt").astype(int)
            df["is_private_room"] = (df["room_type"] == "Private room").astype(int)

        # =========================================
        # 4f. SAFETY SCORE (count of safety amenities)
        # =========================================
        safety_cols = ["has_smoke_alarm", "has_carbon_alarm", "has_fire_extinguisher", "has_first_aid"]
        existing_safety_cols = [c for c in safety_cols if c in df.columns]
        if existing_safety_cols:
            df["safety_amenities_count"] = df[existing_safety_cols].sum(axis=1)

        # =========================================
        # 4g. LOG TRANSFORMS (for skewed features)
        # =========================================
        # These features have long tails that can hurt model performance
        # Note: Applied AFTER ratio/interaction features are computed
        skewed_features = ["number_of_reviews", "host_listings_count"]
        df = apply_log_transforms(df, skewed_features)

        # Log transform price and minimum_nights separately (keep originals for ratios)
        if "price" in df.columns:
            df["log_price"] = np.log1p(df["price"].fillna(0).clip(lower=0))
        if "minimum_nights" in df.columns:
            df["log_minimum_nights"] = np.log1p(df["minimum_nights"].fillna(0).clip(lower=0))

        processed_dfs.append(df)

    if not processed_dfs:
        empty_summary = {
            "original_rows": 0,
            "original_cols": 0,
            "rows_dropped_missing_target": 0,
            "duplicates_removed": 0,
            "final_rows": 0,
            "final_cols": 0,
            "columns_dropped": [],
            "features_created": [],
        }
        return pd.DataFrame(), empty_summary

    # Combine all chunks
    df = pd.concat(processed_dfs, ignore_index=True)
    # =========================================
    # 5. DROP UNNECESSARY COLUMNS
    # =========================================
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    # =========================================
    # 6. DROP ROWS WITH MISSING TARGET
    # =========================================
    rows_before_target_drop = len(df)
    if drop_missing_target and TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])
    rows_dropped_missing_target = rows_before_target_drop - len(df)
    # =========================================
    # 7. REMOVE DUPLICATES
    # =========================================
    rows_before_dedup = len(df)
    if is_training:
        df = df.drop_duplicates()
    duplicates_removed = rows_before_dedup - len(df)
    # Build summary
    new_features = [
        # Original features
        "amenities_count", "description_length", "description_word_count",
        "name_length", "has_neighborhood_overview", "has_host_about",
        "host_about_length", "host_verifications_count",
        # New: Key amenities (binary)
        "has_wifi", "has_kitchen", "has_ac", "has_heating", "has_washer",
        "has_dryer", "has_free_parking", "has_workspace", "has_tv",
        "has_hot_water", "has_self_checkin", "has_smoke_alarm",
        "has_carbon_alarm", "has_fire_extinguisher", "has_first_aid",
        "has_pool", "has_hot_tub", "has_gym", "has_pets_allowed",
        # New: Host & response features
        "host_experience_days", "response_time_score",
        # New: Ratio features
        "price_per_person", "bedrooms_per_person", "beds_per_person",
        "bathrooms_per_person",
        # New: Room type encoding
        "is_entire_home", "is_private_room",
        # New: Safety score
        "safety_amenities_count",
        # New: Text mining (vibe check)
        "luxury_count", "warning_count",
        # New: Interaction feature
        "min_stay_cost",
        # New: Log transforms
        "log_number_of_reviews", "log_host_listings_count",
        "log_price", "log_minimum_nights",
    ]
    cleaning_summary = {
        "original_rows": total_original_rows,
        "original_cols": total_original_cols,
        "rows_dropped_missing_target": rows_dropped_missing_target,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(df),
        "final_cols": len(df.columns),
        "columns_dropped": cols_to_drop,
        "features_created": new_features,
    }
    return df, cleaning_summary



def preprocess(file_inputs:Union[str,pd.DataFrame], drop_missing_target: bool = True,is_training:bool = False):
    if isinstance(file_inputs, str):
        file_inputs = [file_inputs]

    if isinstance(file_inputs, str) or isinstance(file_inputs, list):
        dfs=[]
        for path in file_inputs:
            if not os.path.exists(path):
                print(f"Warning: Path {path} does not exist. Skipping.")
            else:
              dfs.append(pd.read_csv(path))
    else:
        dfs=[file_inputs]
    #     continue
    # 1. Run your existing multi-file cleaning and feature extraction
    df, summary = run_initial_cleaning(dfs, drop_missing_target,is_training)

    # 2. Final Feature Selection
    # Keep categorical strings (like neighborhood) so train.py can encode them
    # We only drop the columns explicitly marked for exclusion
    cols_to_drop = [TARGET_COL] + COLS_TO_EXCLUDE
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    y = df[TARGET_COL] if TARGET_COL in df.columns else None

    if y is not None:
        df_gold = pd.concat([X, y], axis=1)
    else:
        df_gold = X

    summary["final_rows"] = len(df_gold)
    summary["final_cols"] = len(df_gold.columns)
    return df_gold, summary


def main():
    parser = argparse.ArgumentParser(
        description="AirBnB preprocessing for one or multiple files: feature engineering, cleaning, and merging"
    )

    parser.add_argument("input_paths", nargs="+", help="One or more paths to raw AirBnB CSV files")
    parser.add_argument("--out-csv", default=DEFAULT_OUTPUT_CSV, help="Output path for cleaned combined CSV")
    parser.add_argument("--out-summary", default=DEFAULT_SUMMARY_JSON, help="Output path for cleaning summary JSON")
    parser.add_argument(
        "--keep-missing-target",
        action="store_true",
        help="If set, keep rows with missing target (for inference mode)"
    )
    parser.add_argument(
        "--is-training",
        action="store_true",
        help="If set, enables training mode (removes duplicates)"
    )

    args = parser.parse_args()
    print("args: ", args)
    df_clean, summary = preprocess(args.input_paths,
                                   drop_missing_target=not args.keep_missing_target,
                                   is_training=args.is_training
                                   )

    _ensure_parent_dir(args.out_csv)



    df_clean.to_csv(args.out_csv, index=False)

    _ensure_parent_dir(args.out_summary)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Saved cleaned data to: {args.out_csv}")
    print(f"Saved summary to:      {args.out_summary}")
    print()
    print("Summary:")
    print(f"  Original rows:              {summary['original_rows']:,}")
    print(f"  Rows dropped (no target):   {summary['rows_dropped_missing_target']:,}")
    print(f"  Duplicates removed:         {summary['duplicates_removed']:,}")
    print(f"  Final rows:                 {summary['final_rows']:,}")
    print(f"  Final columns:              {summary['final_cols']}")
    print()
    print(f"  New features created:       {len(summary['features_created'])}")
    for feat in summary['features_created']:
        print(f"    - {feat}")


if __name__ == "__main__":
    main()
