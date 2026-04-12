"""
Models Module
=============
Train, evaluate, and save all regression models for predicting
log-transformed player market values.

Models (1-7): sklearn Pipeline with ColumnTransformer preprocessing
Model 8: skrub tabular_learner (handles raw DataFrames with string columns)
"""

import time
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from xgboost import XGBRegressor


# ── Feature column definitions ──────────────────────────────────────────

NUMERIC_FEATURES = [
    "age", "age_squared", "is_peak_age", "height_in_cm",
    "stadium_seats", "league_tier", "champions_league_flag",
    "champions_league_apps",
    "total_goals", "total_assists", "total_minutes", "num_appearances",
    "total_yellow_cards", "total_red_cards",
    "goals_per_90", "assists_per_90", "yellow_cards_per_90", "red_cards_per_90",
    "goal_involvement", "minutes_per_goal_involvement",
    "num_transfers", "highest_previous_fee", "total_transfer_fees",
]

CATEGORICAL_FEATURES = [
    "position_group", "foot", "confederation",
]

# Columns for skrub (pass raw strings — TableVectorizer handles encoding)
SKRUB_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
    "sub_position", "country_of_citizenship", "club_name",
    "domestic_competition_id",
]

TARGET = "log_market_value"


# ── Preprocessing ───────────────────────────────────────────────────────

def build_preprocessor():
    """Build a ColumnTransformer for sklearn models."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ── Model definitions ───────────────────────────────────────────────────

def get_sklearn_models() -> dict:
    """Return dict of model_name → sklearn Pipeline."""
    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Lasso Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", Lasso(alpha=0.1, max_iter=5000)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeRegressor(
                max_depth=10, min_samples_split=20, random_state=42
            )),
        ]),
        "KNN Regressor": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ]),
    }
    return models


def get_skrub_model():
    """Return skrub learner (handles raw DataFrame)."""
    try:
        from skrub import SkrubLearner
        return SkrubLearner("regressor")
    except ImportError:
        from skrub import tabular_learner
        return tabular_learner("regressor")


# ── Data preparation ────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, min_minutes: int = 450, test_size: float = 0.2):
    """
    Prepare train/test splits from the feature matrix.

    - Filters to players with >= min_minutes
    - Drops rows with NaN in key feature columns
    - Splits 80/20, stratified by position_group
    """
    data = df.copy()

    # Filter to sufficient playing time
    data = data[data["total_minutes"] >= min_minutes].copy()

    # Drop rows with NaN in numeric features or target
    data = data.dropna(subset=NUMERIC_FEATURES + [TARGET])

    # For sklearn models: extract X and y
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    # Fill any remaining NaN categoricals
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown")

    y = data[TARGET].values

    # For skrub: keep full raw DataFrame
    X_skrub = data[SKRUB_FEATURES].copy()
    for col in X_skrub.select_dtypes("object").columns:
        X_skrub[col] = X_skrub[col].fillna("Unknown")

    # Stratified split by position_group
    strat_col = X["position_group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat_col,
    )

    # Match skrub splits by index
    X_skrub_train = X_skrub.loc[X_train.index]
    X_skrub_test = X_skrub.loc[X_test.index]

    # Keep metadata for display
    meta_test = data.loc[X_test.index, [
        "player_id", "player_name", "position_group", "club_name",
        "market_value_in_eur", "season",
    ]].copy()

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_skrub_train": X_skrub_train, "X_skrub_test": X_skrub_test,
        "meta_test": meta_test,
    }


# ── Evaluation ──────────────────────────────────────────────────────────

def compute_metrics(y_true_log, y_pred_log, clip_log_min=0.0, clip_log_max=None):
    """
    Compute metrics on both log-space and euro-space.

    Clips predictions to [clip_log_min, clip_log_max] in log-space before
    computing euro-space metrics — prevents linear models from producing
    extreme overestimates that explode after np.expm1().
    """
    # R2 on raw (unclipped) log predictions
    r2 = r2_score(y_true_log, y_pred_log)

    # Clip predictions in log-space for euro metrics
    y_pred_clipped = np.clip(y_pred_log, clip_log_min, clip_log_max)

    # Inverse transform to euro space for interpretable metrics
    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_clipped)

    mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse_eur = root_mean_squared_error(y_true_eur, y_pred_eur)

    # MAPE — avoid division by zero
    mask = y_true_eur > 0
    mape = np.mean(np.abs(
        (y_true_eur[mask] - y_pred_eur[mask]) / y_true_eur[mask]
    )) * 100

    return {
        "R2": round(r2, 4),
        "MAE_EUR": round(mae_eur, 0),
        "RMSE_EUR": round(rmse_eur, 0),
        "MAPE": round(mape, 2),
    }


# ── Training orchestration ──────────────────────────────────────────────

def train_all_models(df: pd.DataFrame,
                     models_dir: str = "models",
                     cv_folds: int = 5) -> pd.DataFrame:
    """
    Train all models, compute test + CV metrics, save .pkl files.

    Returns a DataFrame of results.
    """
    os.makedirs(models_dir, exist_ok=True)

    print("Preparing data...")
    splits = prepare_data(df)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Compute log-space clipping bounds from training data
    log_clip_max = float(y_train.max()) + 0.5  # small margin above observed max
    print(f"  Log target range: {y_train.min():.2f} - {y_train.max():.2f} (clip max={log_clip_max:.2f})")

    # Save the test metadata and predictions for later use
    results = []

    # ── Train sklearn models (1-7) ──
    sklearn_models = get_sklearn_models()

    all_predictions = {}

    for name, pipeline in sklearn_models.items():
        print(f"\nTraining {name}...")
        start = time.time()

        # Fit
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        # Predict
        y_pred = pipeline.predict(X_test)
        all_predictions[name] = y_pred

        # Metrics (clip log-space predictions to training range)
        metrics = compute_metrics(y_test, y_pred, clip_log_max=log_clip_max)
        metrics["Model"] = name
        metrics["Training_Time_s"] = round(train_time, 2)

        # Cross-validation R2
        print(f"  Running {cv_folds}-fold CV...")
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1,
        )
        metrics["CV_R2_Mean"] = round(cv_scores.mean(), 4)
        metrics["CV_R2_Std"] = round(cv_scores.std(), 4)

        results.append(metrics)

        # Save model
        model_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(pipeline, model_path, compress=3)
        print(f"  R2={metrics['R2']:.4f} | MAE=EUR {metrics['MAE_EUR']:,.0f} | "
              f"CV R2={metrics['CV_R2_Mean']:.4f} +/- {metrics['CV_R2_Std']:.4f} | "
              f"Time={train_time:.1f}s")

    # ── Train skrub model (8) ──
    print("\nTraining skrub tabular_learner...")
    try:
        skrub_model = get_skrub_model()
        start = time.time()
        skrub_model.fit(splits["X_skrub_train"], y_train)
        train_time = time.time() - start

        y_pred_skrub = skrub_model.predict(splits["X_skrub_test"])
        all_predictions["skrub tabular_learner"] = y_pred_skrub

        metrics = compute_metrics(y_test, y_pred_skrub, clip_log_max=log_clip_max)
        metrics["Model"] = "skrub tabular_learner"
        metrics["Training_Time_s"] = round(train_time, 2)

        # CV for skrub
        print(f"  Running {cv_folds}-fold CV...")
        cv_scores = cross_val_score(
            skrub_model, splits["X_skrub_train"], y_train,
            cv=cv_folds, scoring="r2", n_jobs=-1,
        )
        metrics["CV_R2_Mean"] = round(cv_scores.mean(), 4)
        metrics["CV_R2_Std"] = round(cv_scores.std(), 4)

        results.append(metrics)

        model_path = os.path.join(models_dir, "skrub_tabular_learner.pkl")
        joblib.dump(skrub_model, model_path, compress=3)
        print(f"  R2={metrics['R2']:.4f} | MAE=EUR {metrics['MAE_EUR']:,.0f} | "
              f"CV R2={metrics['CV_R2_Mean']:.4f} +/- {metrics['CV_R2_Std']:.4f} | "
              f"Time={train_time:.1f}s")
    except Exception as e:
        print(f"  [WARN] skrub failed: {e}")

    # ── Build results DataFrame ──
    results_df = pd.DataFrame(results)
    cols_order = ["Model", "R2", "CV_R2_Mean", "CV_R2_Std",
                  "MAE_EUR", "RMSE_EUR", "MAPE", "Training_Time_s"]
    results_df = results_df[[c for c in cols_order if c in results_df.columns]]
    results_df = results_df.sort_values("R2", ascending=False).reset_index(drop=True)

    # Save results
    results_path = os.path.join(models_dir, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nModel results saved to {results_path}")

    # Save test predictions for later use (predictions page, undervalued players)
    pred_df = splits["meta_test"].copy()
    pred_df["y_true_log"] = y_test
    for model_name, preds in all_predictions.items():
        col = f"pred_{model_name.lower().replace(' ', '_')}"
        pred_df[col] = preds
    pred_path = os.path.join(models_dir, "test_predictions.parquet")
    pred_df.to_parquet(pred_path, index=False)
    print(f"Test predictions saved to {pred_path}")

    # Save feature list for the preprocessing pipeline
    feature_info = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "skrub_features": SKRUB_FEATURES,
        "target": TARGET,
    }
    with open(os.path.join(models_dir, "feature_info.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    return results_df


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Football Player Market Value Predictor - Model Training")
    print("=" * 60)

    df = pd.read_parquet("data/processed/features.parquet")
    print(f"Loaded features: {df.shape}")

    results = train_all_models(df)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
