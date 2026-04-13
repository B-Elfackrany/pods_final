"""
Quick benchmark: train only XGBoost to measure feature impact.
Usage: python benchmark_quick.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from src.models import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, build_preprocessor


def benchmark():
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

    preprocessor = build_preprocessor()
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    y_true_eur = np.expm1(y_test)
    y_pred_eur = np.expm1(y_pred)
    mae = mean_absolute_error(y_true_eur, y_pred_eur)

    print(f"\n{'='*50}")
    print(f"  QUICK BENCHMARK (XGBoost only)")
    print(f"{'='*50}")
    print(f"  Features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")
    print(f"  Numeric features: {NUMERIC_FEATURES}")
    print(f"  Train/Test: {len(X_train)}/{len(X_test)}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  €{mae:,.0f}")
    print(f"  Time: {train_time:.1f}s")
    print(f"{'='*50}\n")

    return r2, mae


if __name__ == "__main__":
    benchmark()
