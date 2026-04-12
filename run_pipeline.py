"""
Run the full data pipeline: load → join → engineer features → save parquet.

Usage:
    python run_pipeline.py
"""

import time
from src.data_loader import build_base_dataset
from src.feature_engineering import engineer_features


def main():
    start = time.time()

    print("=" * 60)
    print("  Football Player Market Value Predictor — Data Pipeline")
    print("=" * 60)

    # Step 1: Build base dataset from raw CSVs
    base_df = build_base_dataset(data_dir="data")

    # Step 2: Engineer features and save parquet
    features_df = engineer_features(base_df, output_dir="data/processed")

    elapsed = time.time() - start
    print(f"\n[OK] Pipeline complete in {elapsed:.1f}s")
    print(f"   Output: data/processed/features.parquet ({len(features_df)} rows)")


if __name__ == "__main__":
    main()
