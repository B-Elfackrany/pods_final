"""
Page 1: Business Case & Data Presentation
==========================================
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Business Case", page_icon="🏠", layout="wide")

st.title("🏠 Business Case & Data Presentation")
st.markdown("### The Problem: How Much Is a Football Player Worth?")

# ── Data scope toggle (Request #10) ──
TOP5_IDS = {"GB1", "ES1", "IT1", "L1", "FR1"}
data_scope = st.sidebar.radio(
    "🏟️ Data Scope",
    ["All Players", "Top 5 Leagues Only", "€10M+ Players Only"],
    index=0,
    key="bc_scope",
)

# ── Business problem ──
st.markdown(
    """
    Football clubs spend **billions** on player transfers every year. In 2023 alone, 
    the global transfer market exceeded **€9.6 billion**. Yet clubs routinely overpay 
    for players or miss undervalued talent hiding in lower leagues.

    **Our goal:** Build a data-driven tool that estimates a player's market value based 
    on performance stats, player profile, and contextual factors — essentially 
    ***Moneyball for football.***

    > *"In football, the difference between a €5M bargain and a €50M flop can make or 
    > break a club's season."*
    """
)

st.divider()

# ── Load data ──
@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")

@st.cache_data
def load_raw_tables():
    tables = {}
    for name in ["players", "appearances", "player_valuations", "games",
                  "competitions", "clubs", "transfers"]:
        tables[name] = pd.read_csv(f"data/{name}.csv", low_memory=False)
    return tables

df_full = load_features()
raw = load_raw_tables()

# Apply scope filter
if data_scope == "Top 5 Leagues Only":
    df = df_full[df_full["domestic_competition_id"].isin(TOP5_IDS)].copy()
elif data_scope == "€10M+ Players Only":
    df = df_full[df_full["market_value_in_eur"] >= 10_000_000].copy()
else:
    df = df_full.copy()

# ── Key stats ──
st.subheader("📈 Dataset at a Glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", f"{df['player_id'].nunique():,}")
c2.metric("Player-Seasons", f"{len(df):,}")
c3.metric("Competitions", f"{raw['competitions']['name'].nunique()}")
c4.metric("Seasons", f"{df['season'].min()}–{df['season'].max()}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Total Appearances", f"{len(raw['appearances']):,}")
c6.metric("Total Transfers", f"{len(raw['transfers']):,}")
c7.metric("Clubs", f"{raw['clubs']['club_id'].nunique()}")
c8.metric("Avg Market Value", f"€{df['market_value_in_eur'].mean():,.0f}")

st.divider()

# ── Entity-Relationship Diagram ──
st.subheader("🔗 Dataset Structure — Entity-Relationship Diagram")
st.markdown(
    """
    Our dataset consists of **7 relational tables** from Transfermarkt, 
    joined into a single feature matrix for modeling.
    """
)

# Mermaid ER diagram
er_diagram = """
```mermaid
erDiagram
    PLAYERS ||--o{ APPEARANCES : "player_id"
    PLAYERS ||--o{ PLAYER_VALUATIONS : "player_id"
    PLAYERS ||--o{ TRANSFERS : "player_id"
    GAMES ||--o{ APPEARANCES : "game_id"
    COMPETITIONS ||--o{ GAMES : "competition_id"
    CLUBS ||--o{ PLAYERS : "current_club_id"
    CLUBS ||--o{ COMPETITIONS : "domestic_competition_id"

    PLAYERS {
        int player_id PK
        string name
        string position
        date date_of_birth
        string foot
        float height_in_cm
    }
    APPEARANCES {
        int game_id FK
        int player_id FK
        int goals
        int assists
        int minutes_played
    }
    PLAYER_VALUATIONS {
        int player_id FK
        date date
        int market_value_in_eur
    }
    GAMES {
        int game_id PK
        string competition_id FK
        int season
    }
    COMPETITIONS {
        string competition_id PK
        string name
        string type
    }
    CLUBS {
        int club_id PK
        string name
        int stadium_seats
    }
    TRANSFERS {
        int player_id FK
        date transfer_date
        float transfer_fee
    }
```
"""
st.markdown(er_diagram)

st.divider()

# ── Sample data ──
st.subheader("🗂️ Processed Feature Matrix (Sample)")
display_cols = [
    "player_name", "season", "age", "position_group", "club_name",
    "total_goals", "total_assists", "num_appearances", "goals_per_90",
    "league_tier", "market_value_in_eur",
]
sample = df[display_cols].sample(10, random_state=42).sort_values(
    "market_value_in_eur", ascending=False
)
sample["market_value_in_eur"] = sample["market_value_in_eur"].apply(
    lambda x: f"€{x:,.0f}"
)
st.dataframe(sample, use_container_width=True, hide_index=True)

st.divider()

# ── Data quality ──
st.subheader("🔍 Data Quality Summary")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Missing Values in Feature Matrix**")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(nulls) > 0:
        st.dataframe(
            pd.DataFrame({"Column": nulls.index, "Missing": nulls.values,
                          "% Missing": (nulls.values / len(df) * 100).round(2)}),
            hide_index=True,
        )
    else:
        st.success("No missing values!")

with col_b:
    st.markdown("**Target Variable Distribution**")
    st.markdown(
        f"""
        - **Min:** €{df['market_value_in_eur'].min():,.0f}
        - **Median:** €{df['market_value_in_eur'].median():,.0f}
        - **Mean:** €{df['market_value_in_eur'].mean():,.0f}
        - **Max:** €{df['market_value_in_eur'].max():,.0f}
        - **Skewness:** {df['market_value_in_eur'].skew():.2f} (heavy right skew → log transform)
        """
    )

st.divider()

# ── Fun facts ──
st.subheader("⭐ Fun Facts from the Data")

most_valuable = df.loc[df["market_value_in_eur"].idxmax()]
most_appearances = df.loc[df["num_appearances"].idxmax()]

f1, f2, f3 = st.columns(3)
with f1:
    st.metric(
        "Most Valuable Player-Season",
        most_valuable["player_name"],
        f"€{most_valuable['market_value_in_eur']:,.0f} ({int(most_valuable['season'])})",
    )
with f2:
    top_goals = df.loc[df["total_goals"].idxmax()]
    st.metric(
        "Most Goals in a Season",
        top_goals["player_name"],
        f"{int(top_goals['total_goals'])} goals ({int(top_goals['season'])})",
    )
with f3:
    st.metric(
        "Most Appearances in a Season",
        most_appearances["player_name"],
        f"{int(most_appearances['num_appearances'])} games ({int(most_appearances['season'])})",
    )

with st.expander("💡 Why Log-Transform the Target?"):
    st.markdown(
        """
        Market values are **heavily right-skewed**: most players are worth under €5M, 
        but a few superstars are valued at €100M+. This skewness violates regression 
        assumptions and causes models to focus on predicting expensive players while 
        ignoring the majority.

        **Solution:** Apply `np.log1p(market_value)` to compress the range and make 
        the distribution more normal. We inverse-transform predictions with `np.expm1()` 
        for display.
        """
    )
