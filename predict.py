# predict.py
import argparse
from pathlib import Path

import pandas as pd
import joblib

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_INPUT_CSV = "data/processed/test.csv"
DEFAULT_OUTPUT_CSV = "results/predictions_test.csv"

TARGET_COL = "review_scores_rating"
COLS_TO_EXCLUDE = ["city"]

# -----------------------------
# Helper
# -----------------------------
def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Prediction function
# -----------------------------
def predict(model_path: str, input_csv: str, output_csv: str):
    # Load model pipeline
    model = joblib.load(model_path)

    # Load input data
    df = pd.read_csv(input_csv)

    # Select features only (exclude target and excluded columns)
    feature_cols = [c for c in df.columns if c not in [TARGET_COL] + COLS_TO_EXCLUDE]
    X = df[feature_cols]

    # Predict
    predictions = model.predict(X)

    # Save results
    output_df = df.copy()
    output_df["prediction"] = predictions

    _ensure_parent_dir(output_csv)
    output_df.to_csv(output_csv, index=False)

    return output_df

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference using trained model on test set.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model artifact")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV for prediction")
    parser.add_argument("--out-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV with predictions")
    args = parser.parse_args()

    output_df = predict(
        model_path=args.model_path,
        input_csv=args.in_csv,
        output_csv=args.out_csv,
    )

    print("✅ Prediction complete")
    print(f"Model used:      {args.model_path}")
    print(f"Input data:      {args.in_csv}")
    print(f"Predictions saved to: {args.out_csv}")
    print(f"Number of predictions: {len(output_df)}")

if __name__ == "__main__":
    main()
