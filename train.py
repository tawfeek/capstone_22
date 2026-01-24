# train.py
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_TRAIN_CSV = "data/processed/train.csv"
DEFAULT_TEST_CSV = "data/processed/test.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_METRICS_PATH = "results/metrics.json"
DEFAULT_PREDICTIONS_CSV = "results/predictions.csv"

TARGET_COL = "review_scores_rating"
COLS_TO_EXCLUDE = ["city"]

# -----------------------------
# Helper
# -----------------------------
def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Preprocessing
# -----------------------------
def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

# -----------------------------
# Model configs
# -----------------------------
def get_model_configs(random_state: int) -> dict:
    return {
        "ridge": {
            "model": Ridge(random_state=random_state),
            "params": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "random_forest": {
            "model": RandomForestRegressor(
                random_state=random_state,
                n_jobs=2,
                max_features="sqrt",
            ),
            "params": {
                "model__n_estimators": [50, 80, 120],
                "model__max_depth": [8, 12, 16],
                "model__min_samples_split": [5, 10],
                "model__min_samples_leaf": [2, 4],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {
                "model__n_estimators": [50, 80, 120],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__min_samples_split": [5, 10],
            },
        },
    }

# -----------------------------
# Training
# -----------------------------
def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str,
    random_state: int,
    cv_folds: int,
    n_iter_search: int,
    tune_hyperparams: bool,
):
    # Split features / target
    X_train = train_df.drop(columns=[TARGET_COL] + COLS_TO_EXCLUDE, errors="ignore")
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL] + COLS_TO_EXCLUDE, errors="ignore")
    y_test = test_df[TARGET_COL] if TARGET_COL in test_df.columns else None

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    print(f"Train rows: {len(X_train):,}, Test rows: {len(X_test):,}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model_configs = get_model_configs(random_state)

    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}")

    model_cfg = model_configs[model_type]
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model_cfg["model"])])

    # -----------------------------
    # Hyperparameter tuning
    # -----------------------------
    if tune_hyperparams and model_cfg["params"]:
        print(f"Tuning hyperparameters ({n_iter_search} iterations, {cv_folds}-fold CV)...")
        search = RandomizedSearchCV(
            pipeline,
            model_cfg["params"],
            n_iter=n_iter_search,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            n_jobs=2,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_rmse_mean = None
        cv_rmse_std = None
    else:
        pipeline.fit(X_train, y_train)
        best_params = {}
        # -----------------------------
        # Cross-validation (train only)
        # -----------------------------
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=2,
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

    # -----------------------------
    # Test evaluation
    # -----------------------------
    test_predictions = pipeline.predict(X_test)
    if y_test is not None:
        test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        dummy_pred = np.full_like(y_test, y_train.mean())
        dummy_rmse = mean_squared_error(y_test, dummy_pred) ** 0.5
        rmse_improvement = (dummy_rmse - test_rmse) / dummy_rmse * 100
    else:
        test_rmse = test_mae = test_r2 = dummy_rmse = rmse_improvement = None

    # -----------------------------
    # Feature importance
    # -----------------------------
    feature_importance = {}
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        for i in top_idx:
            feature_importance[str(feature_names[i])] = float(importances[i])

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "model_type": model_type,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cv_folds": cv_folds,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "dummy_rmse": dummy_rmse,
        "rmse_improvement_vs_dummy": rmse_improvement,
        "best_params": best_params,
    }

    return pipeline, metrics, feature_importance, test_predictions

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train AirBnB regression model locally.")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--model-type", choices=["ridge", "random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--out-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-metrics", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--out-predictions", default=DEFAULT_PREDICTIONS_CSV)
    args = parser.parse_args()

    print("="*60)
    print("LOADING DATA")
    print("="*60)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    pipeline, metrics, feature_importance, test_preds = train_model(
        train_df=train_df,
        test_df=test_df,
        model_type=args.model_type,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_iter_search=args.n_iter,
        tune_hyperparams=not args.no_tune,
    )

    # Save model
    _ensure_parent_dir(args.out_model)
    joblib.dump(pipeline, args.out_model)

    # Save predictions
    pred_df = test_df.copy()
    pred_df["prediction"] = test_preds
    _ensure_parent_dir(args.out_predictions)
    pred_df.to_csv(args.out_predictions, index=False)

    # Save metrics
    _ensure_parent_dir(args.out_metrics)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "feature_importance": feature_importance}, f, indent=2)

    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"CV RMSE:   {metrics['cv_rmse_mean']}")
    print(f"Test RMSE: {metrics['test_rmse']}")
    print(f"Test MAE:  {metrics['test_mae']}")
    print(f"Test R²:   {metrics['test_r2']}")
    print(f"Dummy RMSE:{metrics['dummy_rmse']}")
    print(f"Improvement vs Dummy: {metrics['rmse_improvement_vs_dummy']:.1f}%")
    print()
    print(f"Model saved to:       {args.out_model}")
    print(f"Predictions saved to: {args.out_predictions}")
    print(f"Metrics saved to:     {args.out_metrics}")

if __name__ == "__main__":
    main()