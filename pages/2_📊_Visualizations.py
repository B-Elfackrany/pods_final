"""
Page 2: Data Visualization & Insights
=======================================
Interactive charts with sidebar filters and deep football analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Visualizations", page_icon="📊", layout="wide")

# ── Style ──
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1B3A2D",
    "text.color": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#FAFAFA",
    "ytick.color": "#FAFAFA",
    "axes.edgecolor": "#FAFAFA",
})

st.title("📊 Data Visualization & Insights")

# ── Load data ──
@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")

@st.cache_data
def load_transfers():
    return pd.read_csv("data/transfers.csv", low_memory=False)

df = load_features()
transfers = load_transfers()

# ── Sidebar filters ──
st.sidebar.subheader("🎛️ Filters")
positions = ["All"] + sorted(df["position_group"].dropna().unique().tolist())
selected_pos = st.sidebar.selectbox("Position Group", positions)

season_range = st.sidebar.slider(
    "Season Range",
    int(df["season"].min()), int(df["season"].max()),
    (int(df["season"].min()), int(df["season"].max()))
)

age_range = st.sidebar.slider("Age Range", 16, 40, (16, 40))

filtered = df.copy()
if selected_pos != "All":
    filtered = filtered[filtered["position_group"] == selected_pos]
filtered = filtered[
    (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1]) &
    (filtered["season"] >= season_range[0]) & (filtered["season"] <= season_range[1])
]

st.caption(f"📌 **{len(filtered):,}** player-season records | "
           f"Position: {selected_pos} | Age: {age_range[0]}-{age_range[1]} | "
           f"Seasons: {season_range[0]}-{season_range[1]}")
st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 1: Market Value Distribution
# ═══════════════════════════════════════════════════════════════
st.subheader("1. 💰 Market Value Distribution — Why We Log-Transform")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(filtered["market_value_in_eur"] / 1e6, bins=80, color="#FFD700",
            edgecolor="black", alpha=0.85)
    ax.set_xlabel("Market Value (€ Millions)")
    ax.set_ylabel("Count")
    ax.set_title("Raw Market Value (Right-Skewed)")
    ax.annotate(f"Skew: {filtered['market_value_in_eur'].skew():.1f}",
                xy=(0.7, 0.85), xycoords="axes fraction",
                fontsize=12, color="#FFD700", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0E1117", alpha=0.8))
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(filtered["log_market_value"], bins=80, color="#4CAF50",
            edgecolor="black", alpha=0.85)
    ax.set_xlabel("Log(1 + Market Value)")
    ax.set_ylabel("Count")
    ax.set_title("Log-Transformed (More Normal)")
    ax.annotate(f"Skew: {filtered['log_market_value'].skew():.1f}",
                xy=(0.7, 0.85), xycoords="axes fraction",
                fontsize=12, color="#4CAF50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0E1117", alpha=0.8))
    st.pyplot(fig)
    plt.close()

st.markdown("> 💡 The raw distribution is heavily right-skewed — most players are worth under €5M. "
            "Log-transforming creates a near-normal distribution, dramatically improving model performance.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 2: Value by Position — Violin Plot
# ═══════════════════════════════════════════════════════════════
st.subheader("2. 🎯 Market Value by Position")
fig, ax = plt.subplots(figsize=(10, 5))
order = ["Attack", "Midfield", "Defender", "Goalkeeper"]
order = [p for p in order if p in filtered["position_group"].values]
sns.violinplot(
    data=filtered, x="position_group", y="log_market_value",
    order=order, palette="YlGn", ax=ax, inner="box", cut=0,
)
ax.set_ylabel("Log Market Value")
ax.set_xlabel("Position")
ax.set_title("Market Value Distribution by Position (Log Scale)")
st.pyplot(fig)
plt.close()

# Add summary metrics below
if len(order) > 0:
    pos_cols = st.columns(len(order))
    for i, pos in enumerate(order):
        pos_data = filtered[filtered["position_group"] == pos]
        with pos_cols[i]:
            st.metric(
                pos,
                f"€{pos_data['market_value_in_eur'].median():,.0f}",
                f"{len(pos_data):,} players",
            )

st.markdown("> Attackers command the highest valuations on average. "
            "Goalkeepers have the tightest distribution — consistent but lower valued.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 3: The Age Curve
# ═══════════════════════════════════════════════════════════════
st.subheader("3. 📈 The Age Curve — When Do Players Peak?")
age_avg = filtered.groupby("age")["market_value_in_eur"].agg(["mean", "median", "count"]).reset_index()
age_avg = age_avg[(age_avg["age"] >= 17) & (age_avg["age"] <= 38) & (age_avg["count"] >= 20)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(age_avg["age"], age_avg["mean"] / 1e6,
        marker="o", color="#FFD700", linewidth=2.5, markersize=6, label="Mean")
ax.plot(age_avg["age"], age_avg["median"] / 1e6,
        marker="s", color="#4CAF50", linewidth=2, markersize=5, label="Median", alpha=0.8)
ax.axvspan(25, 29, alpha=0.12, color="gold", label="Peak Age Window (25-29)")

# Annotate peak
peak_age = age_avg.loc[age_avg["mean"].idxmax(), "age"]
peak_val = age_avg["mean"].max() / 1e6
ax.annotate(f"Peak: Age {int(peak_age)}\n€{peak_val:.1f}M",
            xy=(peak_age, peak_val), xytext=(peak_age + 3, peak_val * 1.1),
            arrowprops=dict(arrowstyle="->", color="#FFD700"),
            fontsize=11, color="#FFD700", fontweight="bold")

ax.set_xlabel("Age")
ax.set_ylabel("Market Value (€ Millions)")
ax.set_title("Average Market Value by Age")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close()
st.markdown("> The famous **age curve**: value peaks around **age 27** and drops sharply after 30. "
            "This is why `age_squared` is an important feature — it captures this non-linear decay.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 4: League Premium
# ═══════════════════════════════════════════════════════════════
st.subheader("4. 🏟️ The League Premium — Top 5 vs Rest")

top5_map = {
    "GB1": "🏴 Premier League", "ES1": "🇪🇸 La Liga", "L1": "🇩🇪 Bundesliga",
    "IT1": "🇮🇹 Serie A", "FR1": "🇫🇷 Ligue 1",
}
league_data = filtered[filtered["domestic_competition_id"].isin(top5_map.keys())].copy()
league_data["league_name"] = league_data["domestic_competition_id"].map(top5_map)
league_avg = league_data.groupby("league_name")["market_value_in_eur"].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(league_avg.index, league_avg.values / 1e6, color="#4CAF50", edgecolor="black")
ax.set_xlabel("Average Player Value (€ Millions)")
ax.set_title("Average Player Market Value by Top 5 League")
for bar, val in zip(bars, league_avg.values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"€{val/1e6:.1f}M", va="center", fontsize=11, color="#FFD700", fontweight="bold")
st.pyplot(fig)
plt.close()

# Add non-top-5 comparison
non_top5_avg = filtered[~filtered["domestic_competition_id"].isin(top5_map.keys())]["market_value_in_eur"].mean()
top5_avg = league_data["market_value_in_eur"].mean()
cA, cB = st.columns(2)
cA.metric("Top 5 League Average", f"€{top5_avg:,.0f}")
cB.metric("Other Leagues Average", f"€{non_top5_avg:,.0f}",
          delta=f"{((non_top5_avg - top5_avg) / top5_avg * 100):.0f}%")

st.markdown("> The **Premier League premium** is real — PL players are valued ~2x higher "
            "than equivalent players in other top leagues.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 5: Foot Preference — Does It Matter?
# ═══════════════════════════════════════════════════════════════
st.subheader("5. 🦶 Foot Preference — Does It Affect Value?")

foot_data = filtered[filtered["foot"].isin(["right", "left", "both"])].copy()
col_a, col_b = st.columns([2, 1])

with col_a:
    fig, ax = plt.subplots(figsize=(8, 4))
    foot_avg = foot_data.groupby("foot")["market_value_in_eur"].agg(["mean", "count"])
    foot_avg = foot_avg.sort_values("mean", ascending=True)
    colors = {"right": "#4CAF50", "left": "#FFD700", "both": "#FF6B6B"}
    bar_colors = [colors.get(f, "#999") for f in foot_avg.index]
    bars = ax.barh(foot_avg.index, foot_avg["mean"] / 1e6, color=bar_colors, edgecolor="black")
    ax.set_xlabel("Average Market Value (€ Millions)")
    ax.set_title("Average Value by Preferred Foot")
    for bar, (foot, row) in zip(bars, foot_avg.iterrows()):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"€{row['mean']/1e6:.1f}M ({int(row['count']):,} players)",
                va="center", fontsize=10, color="#FAFAFA")
    st.pyplot(fig)
    plt.close()

with col_b:
    st.markdown(
        """
        #### 💡 Insight
        **Both-footed** players are valued highest! 
        Being two-footed provides tactical versatility 
        that clubs value highly.
        
        Left-footed players command a slight premium 
        over right-footed players — likely because 
        they're rarer and provide squad balance.
        """
    )

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 6: Champions League Effect
# ═══════════════════════════════════════════════════════════════
st.subheader("6. 🏆 The Champions League Premium")

cl_data = filtered.copy()
cl_data["cl_group"] = cl_data["champions_league_apps"].apply(
    lambda x: "0 apps" if x == 0 else ("1-5 apps" if x <= 5 else ("6-10 apps" if x <= 10 else "10+ apps"))
)
cl_order = ["0 apps", "1-5 apps", "6-10 apps", "10+ apps"]
cl_data["cl_group"] = pd.Categorical(cl_data["cl_group"], categories=cl_order, ordered=True)

fig, ax = plt.subplots(figsize=(10, 5))
cl_avg = cl_data.groupby("cl_group", observed=True)["market_value_in_eur"].agg(["mean", "count"]).reset_index()
bars = ax.bar(cl_avg["cl_group"], cl_avg["mean"] / 1e6, color=["#555", "#4CAF50", "#FFD700", "#FF6B35"],
              edgecolor="black")
ax.set_xlabel("Champions League Appearances")
ax.set_ylabel("Average Market Value (€ Millions)")
ax.set_title("The Champions League Premium")
for bar, (_, row) in zip(bars, cl_avg.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"€{row['mean']/1e6:.1f}M\n({int(row['count']):,})",
            ha="center", fontsize=10, color="#FAFAFA")
ax.grid(True, alpha=0.2, axis="y")
st.pyplot(fig)
plt.close()

st.markdown("> Players with **10+ Champions League appearances** are worth ~5x more than "
            "those with none. CL exposure acts as both a quality signal and value accelerator.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 7: Goals/Assists vs Value
# ═══════════════════════════════════════════════════════════════
st.subheader("7. ⚽ Goals & Assists vs Market Value")
plot_data = filtered[filtered["total_minutes"] >= 450].copy()
if len(plot_data) > 5000:
    plot_data = plot_data.sample(5000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter1 = axes[0].scatter(plot_data["goals_per_90"], plot_data["log_market_value"],
                           alpha=0.3, s=10, c=plot_data["log_market_value"], cmap="YlGn")
z1 = np.polyfit(plot_data["goals_per_90"].dropna(),
                plot_data.loc[plot_data["goals_per_90"].notna(), "log_market_value"], 1)
x_line = np.linspace(0, plot_data["goals_per_90"].max(), 100)
axes[0].plot(x_line, np.polyval(z1, x_line), color="red", linewidth=2, label="Trend")
axes[0].set_xlabel("Goals per 90 min")
axes[0].set_ylabel("Log Market Value")
axes[0].set_title("Goals per 90 vs Value")
axes[0].legend()

scatter2 = axes[1].scatter(plot_data["assists_per_90"], plot_data["log_market_value"],
                           alpha=0.3, s=10, c=plot_data["log_market_value"], cmap="YlGn")
z2 = np.polyfit(plot_data["assists_per_90"].dropna(),
                plot_data.loc[plot_data["assists_per_90"].notna(), "log_market_value"], 1)
x_line2 = np.linspace(0, plot_data["assists_per_90"].max(), 100)
axes[1].plot(x_line2, np.polyval(z2, x_line2), color="red", linewidth=2, label="Trend")
axes[1].set_xlabel("Assists per 90 min")
axes[1].set_ylabel("Log Market Value")
axes[1].set_title("Assists per 90 vs Value")
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("> Clear positive correlation, but with significant **variance** — proving that "
            "goals/assists alone don't determine value. Context (league, age, club) matters equally.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 8: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════
st.subheader("8. 🔥 Feature Correlation Heatmap")

corr_cols = [
    "age", "total_goals", "total_assists", "total_minutes", "num_appearances",
    "goals_per_90", "assists_per_90", "league_tier", "stadium_seats",
    "champions_league_apps", "num_transfers", "highest_previous_fee",
    "log_market_value",
]
corr_data = filtered[corr_cols].dropna()
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="YlGn",
    center=0, ax=ax, square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix", fontsize=14)
st.pyplot(fig)
plt.close()

# Highlight top correlations with target
target_corr = corr_matrix["log_market_value"].drop("log_market_value").abs().sort_values(ascending=False)
st.markdown("**Top correlations with market value:**")
top_corr_cols = st.columns(5)
for i, (feat, corr_val) in enumerate(target_corr.head(5).items()):
    with top_corr_cols[i]:
        st.metric(feat.replace("_", " ").title(), f"r = {corr_val:.3f}")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 9: Top 15 Clubs by Squad Value
# ═══════════════════════════════════════════════════════════════
st.subheader("9. 🏟️ Top 15 Clubs by Total Squad Value")

latest_season = filtered["season"].max()
latest = filtered[filtered["season"] == latest_season]
club_val = latest.groupby("club_name")["market_value_in_eur"].sum().nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(club_val.index, club_val.values / 1e6,
               color=plt.cm.YlGn(np.linspace(0.3, 0.9, len(club_val))), edgecolor="black")
ax.set_xlabel("Total Squad Value (€ Millions)")
ax.set_title(f"Top 15 Clubs by Squad Value ({int(latest_season)}/{int(latest_season)+1} Season)")
for bar, val in zip(bars, club_val.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"€{val/1e6:.0f}M", va="center", fontsize=9, color="#FAFAFA")
st.pyplot(fig)
plt.close()

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 10: Market Value Inflation Over Time
# ═══════════════════════════════════════════════════════════════
st.subheader("10. 📅 Market Value Inflation Over Time")

yearly_stats = df.groupby("season")["market_value_in_eur"].agg(["median", "mean", "count"]).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(yearly_stats["season"], yearly_stats["mean"] / 1e6,
         marker="o", color="#FFD700", linewidth=2.5, markersize=7, label="Mean")
ax1.plot(yearly_stats["season"], yearly_stats["median"] / 1e6,
         marker="s", color="#4CAF50", linewidth=2, markersize=5, label="Median")
ax1.fill_between(yearly_stats["season"], yearly_stats["mean"] / 1e6, alpha=0.1, color="#FFD700")
ax1.set_xlabel("Season")
ax1.set_ylabel("Market Value (€ Millions)")
ax1.set_title("Player Market Value Trends Over Time")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.bar(yearly_stats["season"], yearly_stats["count"], alpha=0.15, color="#FAFAFA", label="Player Count")
ax2.set_ylabel("Number of Players", color="#999")
ax2.tick_params(axis="y", labelcolor="#999")

st.pyplot(fig)
plt.close()
st.markdown("> Market values have **inflated steadily** over the past decade, "
            "driven by TV deals, sponsorship revenue, and competition for talent. "
            "The dip in 2020 is likely COVID-related.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Chart 11: Confederation Analysis
# ═══════════════════════════════════════════════════════════════
st.subheader("11. 🌍 Value by Confederation")

conf_data = filtered[filtered["confederation"].notna()].copy()
conf_avg = conf_data.groupby("confederation")["market_value_in_eur"].agg(["mean", "count"]).reset_index()
conf_avg = conf_avg[conf_avg["count"] >= 50].sort_values("mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(conf_avg["confederation"], conf_avg["mean"] / 1e6,
               color=plt.cm.YlGn(np.linspace(0.3, 0.9, len(conf_avg))), edgecolor="black")
ax.set_xlabel("Average Market Value (€ Millions)")
ax.set_title("Average Player Value by Confederation (Players with 50+ records)")
for bar, (_, row) in zip(bars, conf_avg.iterrows()):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"€{row['mean']/1e6:.1f}M ({int(row['count']):,})",
            va="center", fontsize=9, color="#FAFAFA")
st.pyplot(fig)
plt.close()

st.markdown("> **CONMEBOL** (South America) players command high values — reflecting "
            "the continent's role as a talent pipeline for European clubs.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Player Lookup
# ═══════════════════════════════════════════════════════════════
st.subheader("🔎 Player Lookup — Value Trajectory & Future Prediction")

# Load model for predictions
@st.cache_resource
def load_xgb_model():
    model_path = "models/xgboost.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

xgb_pipeline = load_xgb_model()

player_name = st.text_input("Search for a player (e.g., Kylian Mbappe, Erling Haaland)")

if player_name:
    matches = df[df["player_name"].str.contains(player_name, case=False, na=False)]
    if len(matches) == 0:
        st.warning("No player found. Try a different name.")
    else:
        unique_players = matches["player_name"].unique()
        if len(unique_players) > 1:
            selected = st.selectbox("Multiple matches — select one:", unique_players)
            player_data = matches[matches["player_name"] == selected]
        else:
            player_data = matches

        player_data = player_data.sort_values("season")

        # Player info cards
        latest = player_data.iloc[-1]

        # ── Predict next season value ──
        predicted_value = None
        if xgb_pipeline is not None:
            from src.models import NUMERIC_FEATURES, CATEGORICAL_FEATURES

            next_row = latest.copy()
            next_row["age"] = latest["age"] + 1
            next_row["age_squared"] = next_row["age"] ** 2
            next_row["is_peak_age"] = int(25 <= next_row["age"] <= 29)
            next_row["log_prev_season_value"] = np.log1p(latest["market_value_in_eur"])
            # Update age × position interactions
            for pos in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
                col = f"age_x_{pos.lower()}"
                next_row[col] = next_row["age"] * (1 if latest["position_group"] == pos else 0)

            X_pred = pd.DataFrame([next_row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            for col in CATEGORICAL_FEATURES:
                X_pred[col] = X_pred[col].fillna("Unknown")

            log_pred = xgb_pipeline.predict(X_pred)[0]
            predicted_value = np.expm1(log_pred)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Latest Value", f"€{latest['market_value_in_eur']:,.0f}")
        p2.metric("Position", latest["position_group"])
        p3.metric("Age", f"{int(latest['age'])}")
        p4.metric("Club", latest.get("club_name", "N/A"))

        if predicted_value is not None:
            pct_change = (predicted_value - latest["market_value_in_eur"]) / latest["market_value_in_eur"] * 100
            p5, p6 = st.columns(2)
            p5.metric(
                f"Predicted Value (Next Season, Age {int(latest['age'] + 1)})",
                f"€{predicted_value:,.0f}",
                f"{pct_change:+.1f}%",
            )
            p6.markdown(
                f"*Assuming similar performance next season (same goals, assists, "
                f"appearances, league, and club). The model uses this season's market "
                f"value as the prior.*"
            )

        # ── Plot: historical + predicted ──
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(player_data["season"], player_data["market_value_in_eur"] / 1e6,
                marker="o", color="#FFD700", linewidth=2.5, label="Actual")
        ax.fill_between(player_data["season"], player_data["market_value_in_eur"] / 1e6,
                        alpha=0.15, color="#FFD700")

        if predicted_value is not None:
            next_season = int(latest["season"]) + 1
            # Dashed line connecting last actual to prediction
            ax.plot(
                [latest["season"], next_season],
                [latest["market_value_in_eur"] / 1e6, predicted_value / 1e6],
                marker="D", color="#FF6B6B", linewidth=2, linestyle="--",
                markersize=8, label="Predicted",
            )
            ax.annotate(
                f"€{predicted_value/1e6:.1f}M",
                xy=(next_season, predicted_value / 1e6),
                xytext=(next_season + 0.3, predicted_value / 1e6),
                fontsize=11, color="#FF6B6B", fontweight="bold",
            )

        ax.set_xlabel("Season")
        ax.set_ylabel("Market Value (€ Millions)")
        ax.set_title(f"{player_data['player_name'].iloc[0]} — Value Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            player_data[["season", "age", "position_group", "club_name",
                        "total_goals", "total_assists", "num_appearances",
                        "market_value_in_eur"]].rename(
                columns={"market_value_in_eur": "Market Value (€)"}
            ),
            hide_index=True,
            use_container_width=True,
        )
