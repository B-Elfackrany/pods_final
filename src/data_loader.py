"""
Data Loader Module
==================
Loads and joins the 7 key Transfermarkt CSV tables into a single base DataFrame
ready for feature engineering.
"""

import pandas as pd
import numpy as np
import os


def load_raw_tables(data_dir: str = "data") -> dict:
    """Load all required CSV tables with optimized dtypes."""
    tables = {}

    tables["players"] = pd.read_csv(
        os.path.join(data_dir, "players.csv"),
        usecols=[
            "player_id", "name", "sub_position", "position", "foot",
            "height_in_cm", "date_of_birth", "country_of_citizenship",
            "current_club_id",
        ],
        parse_dates=["date_of_birth"],
    )

    tables["appearances"] = pd.read_csv(
        os.path.join(data_dir, "appearances.csv"),
        usecols=[
            "player_id", "game_id", "competition_id", "goals", "assists",
            "minutes_played", "yellow_cards", "red_cards", "date",
            "player_club_id",
        ],
        parse_dates=["date"],
    )

    tables["player_valuations"] = pd.read_csv(
        os.path.join(data_dir, "player_valuations.csv"),
        usecols=[
            "player_id", "date", "market_value_in_eur", "current_club_id",
        ],
        parse_dates=["date"],
    )

    tables["games"] = pd.read_csv(
        os.path.join(data_dir, "games.csv"),
        usecols=["game_id", "competition_id", "season"],
    )

    tables["competitions"] = pd.read_csv(
        os.path.join(data_dir, "competitions.csv"),
        usecols=[
            "competition_id", "name", "type", "country_name", "confederation",
        ],
    )

    tables["clubs"] = pd.read_csv(
        os.path.join(data_dir, "clubs.csv"),
        usecols=[
            "club_id", "name", "stadium_seats", "domestic_competition_id",
        ],
    )

    tables["transfers"] = pd.read_csv(
        os.path.join(data_dir, "transfers.csv"),
        usecols=[
            "player_id", "transfer_date", "transfer_fee", "from_club_id",
            "to_club_id", "transfer_season",
        ],
        parse_dates=["transfer_date"],
    )

    return tables


def _date_to_season(dates: pd.Series) -> pd.Series:
    """Map dates to season year. Jul+ = that year's season; Jan-Jun = prev year."""
    years = dates.dt.year
    months = dates.dt.month
    return np.where(months >= 7, years, years - 1).astype(int)


def aggregate_appearances(appearances: pd.DataFrame, games: pd.DataFrame,
                          competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate appearance stats per player per season.
    Also counts Champions League / Europa League appearances.
    """
    # Get season from games table
    app_with_season = appearances.merge(
        games[["game_id", "season"]],
        on="game_id",
        how="left",
    )

    # Identify CL and EL competition IDs
    cl_el_ids = competitions.loc[
        competitions["name"].str.contains(
            "champions-league|europa-league|uefa-champions|uefa-europa",
            case=False, na=False,
        ),
        "competition_id",
    ].unique()

    # Flag CL/EL appearances
    app_with_season["is_cl_el"] = app_with_season["competition_id"].isin(cl_el_ids).astype(int)

    # Aggregate per player per season
    agg = app_with_season.groupby(["player_id", "season"]).agg(
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_minutes=("minutes_played", "sum"),
        total_yellow_cards=("yellow_cards", "sum"),
        total_red_cards=("red_cards", "sum"),
        num_appearances=("game_id", "count"),
        champions_league_apps=("is_cl_el", "sum"),
        primary_club_id=("player_club_id", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
    ).reset_index()

    return agg


def get_latest_valuation_per_season(valuations: pd.DataFrame,
                                     min_year: int = 2010) -> pd.DataFrame:
    """
    Map each valuation to a season, filter to min_year+, and keep the
    latest valuation per (player_id, season).
    """
    val = valuations.copy()
    val = val.dropna(subset=["market_value_in_eur", "date"])
    val["season"] = _date_to_season(val["date"])
    val = val[val["season"] >= min_year]

    # Keep latest valuation per player per season
    val = val.sort_values("date").groupby(["player_id", "season"]).last().reset_index()

    return val


def aggregate_transfers(transfers: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transfer history per player:
    - num_previous_transfers
    - highest_previous_fee
    - total_transfer_fees
    """
    tf = transfers.copy()
    tf["transfer_fee"] = pd.to_numeric(tf["transfer_fee"], errors="coerce").fillna(0)

    agg = tf.groupby("player_id").agg(
        num_transfers=("transfer_fee", "count"),
        highest_previous_fee=("transfer_fee", "max"),
        total_transfer_fees=("transfer_fee", "sum"),
    ).reset_index()

    return agg


# ---------- Top-5 league competition IDs ----------
TOP5_LEAGUE_CODES = {"GB1", "ES1", "IT1", "L1", "FR1"}


def build_base_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Orchestrate loading, joining, and merging all tables into a single
    base DataFrame ready for feature engineering.
    """
    print("Loading raw tables...")
    tables = load_raw_tables(data_dir)

    print("Aggregating appearances per player per season...")
    app_agg = aggregate_appearances(
        tables["appearances"], tables["games"], tables["competitions"],
    )

    print("Processing valuations...")
    val = get_latest_valuation_per_season(tables["player_valuations"])

    print("Merging valuations with appearances...")
    # Join valuations with aggregated appearances on (player_id, season)
    df = val.merge(app_agg, on=["player_id", "season"], how="inner")

    print("Computing previous season value (lag feature)...")
    # Create a lookup: (player_id, season) → market_value_in_eur
    prev_val = val[["player_id", "season", "market_value_in_eur"]].copy()
    prev_val["season"] = prev_val["season"] + 1  # shift forward so season N holds value from season N-1
    prev_val = prev_val.rename(columns={"market_value_in_eur": "prev_season_value"})
    df = df.merge(prev_val[["player_id", "season", "prev_season_value"]],
                  on=["player_id", "season"], how="left")
    df["prev_season_value"] = df["prev_season_value"].fillna(0)

    print("Joining player profiles...")
    players = tables["players"].copy()
    players = players.rename(columns={
        "name": "player_name",
        "position": "position_group",
        "current_club_id": "player_current_club_id",
    })
    df = df.merge(
        players[["player_id", "player_name", "sub_position", "position_group",
                 "foot", "height_in_cm", "date_of_birth", "country_of_citizenship"]],
        on="player_id",
        how="left",
    )

    print("Joining club info...")
    clubs = tables["clubs"].rename(columns={"club_id": "current_club_id", "name": "club_name"})
    df = df.merge(
        clubs[["current_club_id", "club_name", "stadium_seats", "domestic_competition_id"]],
        on="current_club_id",
        how="left",
    )

    print("Computing league tier...")
    df["league_tier"] = df["domestic_competition_id"].isin(TOP5_LEAGUE_CODES).astype(int)

    print("Joining competition confederation info...")
    # Map primary_club_id to its domestic competition to get confederation
    comp = tables["competitions"][["competition_id", "confederation"]].drop_duplicates()
    df = df.merge(
        comp.rename(columns={"competition_id": "domestic_competition_id"}),
        on="domestic_competition_id",
        how="left",
    )

    print("Aggregating transfers...")
    tf_agg = aggregate_transfers(tables["transfers"])
    df = df.merge(tf_agg, on="player_id", how="left")
    # Fill NaN for players with no transfer records
    for col in ["num_transfers", "highest_previous_fee", "total_transfer_fees"]:
        df[col] = df[col].fillna(0)

    print(f"Base dataset built: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
