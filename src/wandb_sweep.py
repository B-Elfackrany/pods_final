"""
W&B Hyperparameter Sweep for XGBoost
=====================================
Runs a Bayesian optimization sweep using Weights & Biases.

Usage:
    python -m src.wandb_sweep
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from src.models import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, build_preprocessor, compute_metrics
)

# Load .env for API key
load_dotenv()

import wandb


def prepare_sweep_data():
    """Load and prepare data for sweep."""
    df = pd.read_parquet("data/processed/features.parquet")
    data = df[df["total_minutes"] >= 450].dropna(subset=NUMERIC_FEATURES + [TARGET]).copy()

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown")
    y = data[TARGET].values

    strat_col = X["position_group"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat_col,
    )
    return X_train, X_test, y_train, y_test


def train_sweep():
    """Training function called by wandb agent."""
    X_train, X_test, y_train, y_test = prepare_sweep_data()
    log_clip_max = float(y_train.max()) + 0.5

    with wandb.init() as run:
        config = wandb.config

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.get("colsample_bytree", 0.8),
                min_child_weight=config.get("min_child_weight", 1),
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, clip_log_max=log_clip_max)

        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1,
        )

        wandb.log({
            "r2": metrics["R2"],
            "mae_eur": metrics["MAE_EUR"],
            "rmse_eur": metrics["RMSE_EUR"],
            "mape": metrics["MAPE"],
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "training_time_s": round(train_time, 2),
        })


def run_sweep(count: int = 30):
    """Run W&B sweep with Bayesian optimization."""
    sweep_config = {
        "method": "bayes",
        "name": "xgboost-sweep",
        "metric": {"name": "r2", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]},
            "max_depth": {"values": [3, 4, 5, 6, 7, 8, 10]},
            "n_estimators": {"values": [100, 200, 300, 400, 500]},
            "subsample": {"values": [0.6, 0.7, 0.8, 0.9, 1.0]},
            "colsample_bytree": {"values": [0.6, 0.7, 0.8, 0.9, 1.0]},
            "min_child_weight": {"values": [1, 3, 5, 7]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="football-value-predictor")
    print(f"Sweep ID: {sweep_id}")
    print(f"Running {count} trials...")

    wandb.agent(sweep_id, function=train_sweep, count=count)

    # Fetch best run
    api = wandb.Api()
    sweep = api.sweep(f"football-value-predictor/{sweep_id}")
    best_run = sweep.best_run()

    best_config = best_run.config
    best_metrics = {k: v for k, v in best_run.summary.items()
                    if k in ["r2", "mae_eur", "rmse_eur", "mape", "cv_r2_mean", "cv_r2_std"]}

    # Save results for the Streamlit page
    sweep_results = {
        "sweep_id": sweep_id,
        "n_trials": count,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "sweep_url": best_run.url.rsplit("/runs", 1)[0] if best_run.url else "",
    }

    results_path = "models/wandb_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"\nSweep results saved to {results_path}")
    print(f"Best R²: {best_metrics.get('r2', 'N/A')}")
    print(f"Best config: {best_config}")

    return sweep_results


def main():
    print("=" * 60)
    print("  W&B Hyperparameter Sweep — XGBoost")
    print("=" * 60)

    wandb.login()
    results = run_sweep(count=30)

    print("\n" + "=" * 60)
    print("  SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
