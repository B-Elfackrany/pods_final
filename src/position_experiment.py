"""
Position Encoding Experiment
=============================
Tests whether position-specific feature interactions improve the model
compared to plain one-hot encoding of positions.

Hypothesis: Clean sheets matter for goalkeepers/defenders, goals/assists
matter more for attackers — encoding this explicitly should help.

Usage:
    python -m src.position_experiment
"""

import time
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from src.models import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, build_preprocessor,
    compute_metrics,
)


def create_position_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create position-specific interaction features.
    Instead of one-hot encoding position and letting the model figure out
    that goals matter for attackers but not for GKs, we encode this directly.
    """
    df = df.copy()

    # Goals-related features × position
    df["goals_x_attacker"] = df["goals_per_90"] * (df["position_group"] == "Attack").astype(float)
    df["goals_x_midfielder"] = df["goals_per_90"] * (df["position_group"] == "Midfield").astype(float)
    df["goals_x_defender"] = df["goals_per_90"] * (df["position_group"] == "Defender").astype(float)

    # Assists × midfield (playmaker proxy)
    df["assists_x_midfielder"] = df["assists_per_90"] * (df["position_group"] == "Midfield").astype(float)
    df["assists_x_attacker"] = df["assists_per_90"] * (df["position_group"] == "Attack").astype(float)

    # Clean sheet proxy for GK/Def: low goals per 90 = good for defenders
    # We use negative of goals_per_90 for defenders as a defensive quality proxy
    df["defensive_quality_def"] = (1 - df["goals_per_90"].clip(0, 1)) * (df["position_group"] == "Defender").astype(float)
    df["defensive_quality_gk"] = (1 - df["goals_per_90"].clip(0, 1)) * (df["position_group"] == "Goalkeeper").astype(float)

    # Minutes × position (playing time importance varies)
    df["minutes_x_gk"] = df["total_minutes"] * (df["position_group"] == "Goalkeeper").astype(float)

    # Cards × defender (aggressive defending)
    df["yellows_x_defender"] = df["yellow_cards_per_90"] * (df["position_group"] == "Defender").astype(float)

    return df


def run_experiment():
    """Run the position encoding experiment and compare results."""
    print("Loading data...")
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

    log_clip_max = float(y_train.max()) + 0.5

    results = {}

    # ── Experiment 1: Baseline (current approach — one-hot position) ──
    print("\n1. Baseline: One-hot position encoding...")
    baseline_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])

    start = time.time()
    baseline_pipeline.fit(X_train, y_train)
    baseline_time = time.time() - start
    y_pred_baseline = baseline_pipeline.predict(X_test)
    baseline_metrics = compute_metrics(y_test, y_pred_baseline, clip_log_max=log_clip_max)

    cv_baseline = cross_val_score(baseline_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    results["baseline_onehot"] = {
        **baseline_metrics,
        "cv_r2_mean": round(cv_baseline.mean(), 4),
        "cv_r2_std": round(cv_baseline.std(), 4),
        "training_time": round(baseline_time, 2),
        "description": "Standard one-hot encoding for position_group (current approach)",
    }
    print(f"   R² = {baseline_metrics['R2']:.4f}, CV R² = {cv_baseline.mean():.4f}")

    # ── Experiment 2: Position interaction features ──
    print("\n2. Position-specific interaction features...")
    data_with_interactions = create_position_interaction_features(data)

    interaction_cols = [
        "goals_x_attacker", "goals_x_midfielder", "goals_x_defender",
        "assists_x_midfielder", "assists_x_attacker",
        "defensive_quality_def", "defensive_quality_gk",
        "minutes_x_gk", "yellows_x_defender",
    ]

    # Numeric features + interaction features
    enhanced_numeric = NUMERIC_FEATURES + interaction_cols

    X_enhanced = data_with_interactions[enhanced_numeric + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_enhanced[col] = X_enhanced[col].fillna("Unknown")

    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=X_enhanced["position_group"],
    )

    enhanced_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), enhanced_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    enhanced_pipeline = Pipeline([
        ("preprocessor", enhanced_preprocessor),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])

    start = time.time()
    enhanced_pipeline.fit(Xe_train, ye_train)
    enhanced_time = time.time() - start
    y_pred_enhanced = enhanced_pipeline.predict(Xe_test)
    enhanced_metrics = compute_metrics(ye_test, y_pred_enhanced, clip_log_max=log_clip_max)

    cv_enhanced = cross_val_score(enhanced_pipeline, Xe_train, ye_train, cv=5, scoring="r2", n_jobs=-1)
    results["position_interactions"] = {
        **enhanced_metrics,
        "cv_r2_mean": round(cv_enhanced.mean(), 4),
        "cv_r2_std": round(cv_enhanced.std(), 4),
        "training_time": round(enhanced_time, 2),
        "description": "One-hot + position-specific interaction features (goals_x_attacker, etc.)",
        "extra_features": interaction_cols,
    }
    print(f"   R² = {enhanced_metrics['R2']:.4f}, CV R² = {cv_enhanced.mean():.4f}")

    # ── Experiment 3: No position encoding (drop position entirely) ──
    print("\n3. No position encoding (remove position_group)...")
    no_pos_cats = [c for c in CATEGORICAL_FEATURES if c != "position_group"]
    no_pos_numeric = [c for c in NUMERIC_FEATURES
                      if not c.startswith("age_x_")]  # also remove age_x_position interactions

    no_pos_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), no_pos_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), no_pos_cats),
        ],
        remainder="drop",
    )

    Xnp_train = X_train[no_pos_numeric + no_pos_cats].copy()
    Xnp_test = X_test[no_pos_numeric + no_pos_cats].copy()

    no_pos_pipeline = Pipeline([
        ("preprocessor", no_pos_preprocessor),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])

    start = time.time()
    no_pos_pipeline.fit(Xnp_train, y_train)
    no_pos_time = time.time() - start
    y_pred_no_pos = no_pos_pipeline.predict(Xnp_test)
    no_pos_metrics = compute_metrics(y_test, y_pred_no_pos, clip_log_max=log_clip_max)

    cv_no_pos = cross_val_score(no_pos_pipeline, Xnp_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    results["no_position"] = {
        **no_pos_metrics,
        "cv_r2_mean": round(cv_no_pos.mean(), 4),
        "cv_r2_std": round(cv_no_pos.std(), 4),
        "training_time": round(no_pos_time, 2),
        "description": "Position removed entirely — model uses only stats and context",
    }
    print(f"   R² = {no_pos_metrics['R2']:.4f}, CV R² = {cv_no_pos.mean():.4f}")

    # ── Save results ──
    output_path = "models/position_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  POSITION ENCODING EXPERIMENT — SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"\n  {name}:")
        print(f"    R² = {res['R2']:.4f} | CV R² = {res['cv_r2_mean']:.4f} | MAE = €{res['MAE_EUR']:,.0f}")
        print(f"    {res['description']}")

    return results


def main():
    print("=" * 60)
    print("  Position Encoding Experiment")
    print("=" * 60)
    run_experiment()


if __name__ == "__main__":
    main()
