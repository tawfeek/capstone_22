# results.py
import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PREDICTIONS_CSV = "results/predictions_test.csv"
DEFAULT_RESULTS_JSON = "results/prediction_summary.json"


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def summarize_predictions(pred_csv: str, out_json: str):
    df = pd.read_csv(pred_csv)

    summary = {
        "n_predictions": int(len(df)),
        "prediction_min": float(df["prediction"].min()),
        "prediction_max": float(df["prediction"].max()),
        "prediction_mean": float(df["prediction"].mean()),
        "prediction_std": float(df["prediction"].std()),
    }

    _ensure_parent_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize model predictions.")
    parser.add_argument("--pred-csv", default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--out-json", default=DEFAULT_RESULTS_JSON)
    args = parser.parse_args()

    summary = summarize_predictions(args.pred_csv, args.out_json)

    print("✅ Results summary complete")
    print("Summary:", summary)


if __name__ == "__main__":
    main()