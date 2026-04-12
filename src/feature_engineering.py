"""
Feature Engineering Module
==========================
Creates all derived features from the base dataset produced by data_loader.
Outputs the final feature matrix saved as parquet.
"""

import pandas as pd
import numpy as np
import os


# ── Confederation mapping (Transfermarkt lowercase → standard names) ──
CONFEDERATION_MAP = {
    "europa": "UEFA",
    "asien": "AFC",
    "afrika": "CAF",
    "amerika": "CONMEBOL/CONCACAF",
    "nordamerika": "CONCACAF",
    "südamerika": "CONMEBOL",
}


def compute_per90_stats(df: pd.DataFrame, min_minutes: int = 450) -> pd.DataFrame:
    """
    Compute per-90-minute stats.  Only meaningful for players with >= min_minutes.
    Players below the threshold get 0.
    """
    df = df.copy()

    mask = df["total_minutes"] >= min_minutes

    for raw_col, per90_col in [
        ("total_goals", "goals_per_90"),
        ("total_assists", "assists_per_90"),
        ("total_yellow_cards", "yellow_cards_per_90"),
        ("total_red_cards", "red_cards_per_90"),
    ]:
        df[per90_col] = 0.0
        df.loc[mask, per90_col] = (
            df.loc[mask, raw_col] / df.loc[mask, "total_minutes"] * 90
        )

    return df


def compute_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute age-related and profile features."""
    df = df.copy()

    # Age at valuation
    df["birth_year"] = df["date_of_birth"].dt.year
    # season represents the starting year of the season (e.g. 2020 = 2020/21)
    df["age"] = df["season"] - df["birth_year"]
    # Clip unreasonable ages
    df["age"] = df["age"].clip(15, 45)
    df["age_squared"] = df["age"] ** 2
    df["is_peak_age"] = ((df["age"] >= 25) & (df["age"] <= 29)).astype(int)

    # Position group — already exists in column 'position_group'
    # The dataset uses 'Missing' for ~100 rows with no position; we'll drop these later
    df["position_group"] = df["position_group"].replace({"Missing": np.nan})

    # Foot — fill missing
    df["foot"] = df["foot"].fillna("unknown")

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute goal involvement and related derived stats."""
    df = df.copy()

    df["goal_involvement"] = df["total_goals"] + df["total_assists"]

    # Minutes per goal involvement (capped at 9999 if 0 involvement)
    df["minutes_per_goal_involvement"] = np.where(
        df["goal_involvement"] > 0,
        df["total_minutes"] / df["goal_involvement"],
        9999.0,
    )

    # Champions league flag (1 if any CL/EL appearances)
    df["champions_league_flag"] = (df["champions_league_apps"] > 0).astype(int)

    return df


def normalize_confederation(df: pd.DataFrame) -> pd.DataFrame:
    """Map Transfermarkt confederation names to standard abbreviations."""
    df = df.copy()
    df["confederation"] = (
        df["confederation"]
        .str.lower()
        .map(CONFEDERATION_MAP)
        .fillna("Other")
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle remaining missing values."""
    df = df.copy()

    # Height: fill with median by position group
    if df["height_in_cm"].isnull().any():
        medians = df.groupby("position_group")["height_in_cm"].transform("median")
        df["height_in_cm"] = df["height_in_cm"].fillna(medians)
        # If still NaN (entire position group missing), fill with global median
        df["height_in_cm"] = df["height_in_cm"].fillna(df["height_in_cm"].median())

    # Stadium seats: fill with 0
    df["stadium_seats"] = df["stadium_seats"].fillna(0)

    # Transfer-related: already filled in data_loader, but double-check
    for col in ["num_transfers", "highest_previous_fee", "total_transfer_fees"]:
        df[col] = df[col].fillna(0)

    return df


def engineer_features(base_df: pd.DataFrame,
                      output_dir: str = "data/processed") -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    base_df : pd.DataFrame
        The base dataset from data_loader.build_base_dataset()
    output_dir : str
        Directory to save the output parquet.

    Returns
    -------
    pd.DataFrame
        The engineered feature matrix.
    """
    print("Starting feature engineering...")

    df = base_df.copy()

    # Drop rows with no market value
    before = len(df)
    df = df.dropna(subset=["market_value_in_eur"])
    df = df[df["market_value_in_eur"] > 0]
    print(f"  Dropped {before - len(df)} rows with missing/zero market value")

    # Step 1: Per-90 stats
    print("  Computing per-90 stats...")
    df = compute_per90_stats(df)

    # Step 2: Player profile features
    print("  Computing player features...")
    df = compute_player_features(df)

    # Drop rows with no valid position group (~100 rows)
    before_pos = len(df)
    df = df.dropna(subset=["position_group"])
    dropped_pos = before_pos - len(df)
    if dropped_pos > 0:
        print(f"  Dropped {dropped_pos} rows with missing position group")

    # Step 3: Derived features
    print("  Computing derived features...")
    df = compute_derived_features(df)

    # Step 4: Normalize confederation
    print("  Normalizing confederations...")
    df = normalize_confederation(df)

    # Step 5: Handle missing values
    print("  Handling missing values...")
    df = handle_missing_values(df)

    # Step 6: Log-transform target
    df["log_market_value"] = np.log1p(df["market_value_in_eur"])

    # Select and order final columns
    final_columns = [
        # Identifiers
        "player_id", "player_name", "season",
        # Player profile
        "age", "age_squared", "is_peak_age", "position_group", "sub_position",
        "foot", "height_in_cm", "country_of_citizenship", "confederation",
        # Club / league context
        "club_name", "stadium_seats", "league_tier",
        "domestic_competition_id", "champions_league_flag", "champions_league_apps",
        # Appearance stats (raw)
        "total_goals", "total_assists", "total_minutes", "num_appearances",
        "total_yellow_cards", "total_red_cards",
        # Per-90 stats
        "goals_per_90", "assists_per_90", "yellow_cards_per_90", "red_cards_per_90",
        # Derived
        "goal_involvement", "minutes_per_goal_involvement",
        # Transfer history
        "num_transfers", "highest_previous_fee", "total_transfer_fees",
        # Target
        "market_value_in_eur", "log_market_value",
    ]

    # Only keep columns that exist
    final_columns = [c for c in final_columns if c in df.columns]
    df = df[final_columns]

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nFeature matrix saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Market value range: €{df['market_value_in_eur'].min():,.0f} – €{df['market_value_in_eur'].max():,.0f}")
    print(f"  Seasons: {sorted(df['season'].unique())[:3]} ... {sorted(df['season'].unique())[-3:]}")
    print(f"  Position groups: {df['position_group'].value_counts().to_dict()}")

    return df
