"""
SHAP Analysis Module
====================
Pre-compute SHAP values for the XGBoost model and save as pickle.
Load in Streamlit pages — never compute live.
"""

import os
import pickle
import numpy as np
import pandas as pd
import joblib
import shap

from src.models import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def compute_shap_values(models_dir: str = "models",
                         data_path: str = "data/processed/features.parquet",
                         output_dir: str = "models",
                         sample_size: int = 2000):
    """
    Pre-compute SHAP values for XGBoost and save as pickle.

    Parameters
    ----------
    sample_size : int
        Number of samples for SHAP computation (full dataset is too slow).
    """
    print("Loading data and model...")
    df = pd.read_parquet(data_path)

    # Filter and prepare data
    data = df[df["total_minutes"] >= 450].dropna(subset=NUMERIC_FEATURES + [TARGET]).copy()
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown")

    # Load XGBoost pipeline
    xgb_pipeline = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    xgb_model = xgb_pipeline.named_steps["model"]

    # Transform features
    X_transformed = preprocessor.transform(X)

    # Get feature names after preprocessing
    num_names = NUMERIC_FEATURES
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = num_names + cat_names

    # Sample for SHAP (full dataset is too slow)
    np.random.seed(42)
    idx = np.random.choice(len(X_transformed), size=min(sample_size, len(X_transformed)), replace=False)
    X_sample = X_transformed[idx]

    # Also keep metadata for the sampled rows
    meta_cols = ["player_id", "player_name", "position_group", "club_name",
                 "season", "market_value_in_eur"]
    meta_sample = data.iloc[idx][meta_cols].reset_index(drop=True)

    print(f"Computing SHAP values for {len(X_sample)} samples...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)

    # Save everything
    shap_data = {
        "shap_values": shap_values,
        "X_sample": X_sample,
        "feature_names": all_feature_names,
        "meta_sample": meta_sample,
        "expected_value": explainer.expected_value,
    }

    output_path = os.path.join(output_dir, "shap_values.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(shap_data, f)
    print(f"SHAP values saved to {output_path}")

    # Also compute per-position SHAP for position-specific insights
    print("Computing position-specific SHAP...")
    position_shap = {}
    for pos in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
        pos_mask = meta_sample["position_group"] == pos
        if pos_mask.sum() > 50:
            pos_idx = np.where(pos_mask)[0]
            X_pos = X_sample[pos_idx]
            shap_pos = explainer(X_pos)
            position_shap[pos] = {
                "shap_values": shap_pos,
                "X_sample": X_pos,
                "n_samples": len(pos_idx),
            }
            print(f"  {pos}: {len(pos_idx)} samples")

    pos_path = os.path.join(output_dir, "shap_by_position.pkl")
    with open(pos_path, "wb") as f:
        pickle.dump(position_shap, f)
    print(f"Position-specific SHAP saved to {pos_path}")

    return shap_data


def main():
    print("=" * 60)
    print("  Pre-computing SHAP values")
    print("=" * 60)
    compute_shap_values()
    print("\n[OK] SHAP computation complete")


if __name__ == "__main__":
    main()
