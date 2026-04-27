"""
Page 3: Predictions
====================
Model comparison, interactive prediction tool, and undervalued player finder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import joblib
import json
import os

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1B3A2D",
    "text.color": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#FAFAFA",
    "ytick.color": "#FAFAFA",
    "axes.edgecolor": "#FAFAFA",
})

st.title("🔮 Predictions")

# ── Data scope toggle ──
TOP5_IDS = {"GB1", "ES1", "IT1", "L1", "FR1"}
data_scope = st.sidebar.radio(
    "🏟️ Data Scope",
    ["All Players", "Top 5 Leagues Only", "€10M+ Players Only"],
    index=0,
    key="pred_scope",
)

# ── Load artifacts ──
@st.cache_data
def load_results():
    return pd.read_csv("models/model_results.csv")

@st.cache_data
def load_predictions():
    return pd.read_parquet("models/test_predictions.parquet")

@st.cache_resource
def load_model(name):
    path = os.path.join("models", f"{name.lower().replace(' ', '_')}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_feature_info():
    with open("models/feature_info.json") as f:
        return json.load(f)

@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")

results = load_results()
predictions_raw = load_predictions()
feature_info = load_feature_info()

# Apply scope filter to predictions
predictions = predictions_raw.copy()
if data_scope == "Top 5 Leagues Only":
    df_full = load_features()
    top5_players = df_full[df_full["domestic_competition_id"].isin(TOP5_IDS)]["player_id"].unique()
    if "player_id" in predictions.columns:
        predictions = predictions[predictions["player_id"].isin(top5_players)]
elif data_scope == "€10M+ Players Only":
    if "market_value_in_eur" in predictions.columns:
        predictions = predictions[predictions["market_value_in_eur"] >= 10_000_000]

# Build model MAE lookup for error estimates
model_mae = dict(zip(results["Model"], results["MAE_EUR"]))

# ═══════════════════════════════════════════════════════════════
# Section 1: Model Comparison
# ═══════════════════════════════════════════════════════════════
# ── Data split info ──
df_features = load_features()
total_eligible = len(df_features[df_features["total_minutes"] >= 450].dropna(
    subset=["log_market_value"]))
test_size = len(predictions_raw)
train_size = total_eligible - test_size
train_pct = train_size / total_eligible * 100
test_pct = test_size / total_eligible * 100

split_cols = st.columns(4)
split_cols[0].metric("Total Eligible Records", f"{total_eligible:,}")
split_cols[1].metric("Training Set", f"{train_size:,} ({train_pct:.0f}%)")
split_cols[2].metric("Test Set", f"{test_size:,} ({test_pct:.0f}%)")
split_cols[3].metric("Split Strategy", "80/20 Stratified")
st.caption("Split is stratified by position group to ensure balanced representation. "
           "Only players with 450+ minutes are included.")
st.divider()

st.subheader("📊 Model Comparison")

# Format results for display
display_results = results.copy()
display_results["MAE (€)"] = display_results["MAE_EUR"].apply(lambda x: f"€{x:,.0f}")
display_results["RMSE (€)"] = display_results["RMSE_EUR"].apply(lambda x: f"€{x:,.0f}")
display_results["MAPE (%)"] = display_results["MAPE"].apply(lambda x: f"{x:.1f}%")
display_results["R²"] = display_results["R2"].apply(lambda x: f"{x:.4f}")
display_results["CV R²"] = display_results["CV_R2_Mean"].apply(lambda x: f"{x:.4f}")
display_results["Time (s)"] = display_results["Training_Time_s"].apply(lambda x: f"{x:.2f}")

st.dataframe(
    display_results[["Model", "R²", "CV R²", "MAE (€)", "RMSE (€)", "MAPE (%)", "Time (s)"]],
    hide_index=True,
    use_container_width=True,
)

# R² bar chart
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#FFD700" if r == results["R2"].max() else "#4CAF50" for r in results["R2"]]
    bars = ax.barh(results["Model"], results["R2"], color=colors, edgecolor="black")
    ax.set_xlabel("R² Score")
    ax.set_title("Model Comparison — R² Score")
    ax.set_xlim(0.5, 0.95)
    for bar, val in zip(bars, results["R2"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, color="#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_mae = ["#FFD700" if m == results["MAE_EUR"].min() else "#4CAF50" for m in results["MAE_EUR"]]
    bars = ax.barh(results["Model"], results["MAE_EUR"] / 1e6, color=colors_mae, edgecolor="black")
    ax.set_xlabel("MAE (€ Millions)")
    ax.set_title("Model Comparison — Mean Absolute Error")
    for bar, val in zip(bars, results["MAE_EUR"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"€{val/1e6:.1f}M", va="center", fontsize=10, color="#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ═══════════════════════════════════════════════════════════════
# Section 2: Predicted vs Actual Scatter
# ═══════════════════════════════════════════════════════════════
st.subheader("📈 Predicted vs Actual")

pred_cols = [c for c in predictions.columns if c.startswith("pred_")]
model_names = [c.replace("pred_", "").replace("_", " ").title() for c in pred_cols]
model_map = dict(zip(model_names, pred_cols))

selected_model = st.selectbox("Select Model", model_names, index=0)
pred_col = model_map[selected_model]

col_a, col_b = st.columns(2)
with col_a:
    y_true_eur = np.expm1(predictions["y_true_log"])
    y_pred_eur = np.expm1(predictions[pred_col])
    max_val = max(y_true_eur.max(), y_pred_eur.max()) / 1e6
    axis_max = min(max_val, 200)

    # Build hover text with player info
    hover_texts = []
    for idx in predictions.index:
        row = predictions.loc[idx]
        name = row.get("player_name", "Unknown")
        season = row.get("season", None)
        actual = np.expm1(row["y_true_log"])
        predicted = np.expm1(row[pred_col])
        season_str = f"Season: {int(season)}/{int(season)+1}<br>" if pd.notna(season) else ""
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"{season_str}"
            f"Actual: €{actual/1e6:.1f}M<br>"
            f"Predicted: €{predicted/1e6:.1f}M<br>"
            f"Diff: €{(predicted - actual)/1e6:+.1f}M"
        )

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scattergl(
        x=y_true_eur / 1e6, y=y_pred_eur / 1e6,
        mode="markers",
        marker=dict(size=4, color="#FFD700", opacity=0.4),
        text=hover_texts,
        hoverinfo="text",
        name="Players",
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[0, axis_max], y=[0, axis_max],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Perfect Prediction",
        hoverinfo="skip",
    ))
    fig_scatter.update_layout(
        title=f"{selected_model} — Predicted vs Actual",
        xaxis_title="Actual Value (€ Millions)",
        yaxis_title="Predicted Value (€ Millions)",
        xaxis=dict(range=[0, axis_max]),
        yaxis=dict(range=[0, axis_max]),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1B3A2D",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    # Residuals distribution
    residuals_eur = (y_pred_eur - y_true_eur) / 1e6
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(residuals_eur.clip(-50, 50), bins=80, color="#4CAF50", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (€ Millions)")
    ax.set_ylabel("Count")
    ax.set_title(f"{selected_model} — Residuals Distribution")

    # Add error stats
    mae_val = np.mean(np.abs(residuals_eur))
    median_err = np.median(np.abs(residuals_eur))
    ax.annotate(f"MAE: €{mae_val:.1f}M\nMedian: €{median_err:.1f}M",
                xy=(0.7, 0.85), xycoords="axes fraction",
                fontsize=11, color="#FFD700", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0E1117", alpha=0.8))
    st.pyplot(fig)
    plt.close()

with st.expander("💡 Why Linear Models Have Higher RMSE"):
    st.markdown(
        """
        **Unbounded linear models predicted a player worth €2.9 billion** — this is why we 
        clip predictions and why non-linear models matter. Linear regression extrapolates 
        linearly into extreme values, but tree-based models (Random Forest, XGBoost) are 
        naturally bounded by the training data range.

        We clip log-space predictions to the training data maximum + margin to prevent 
        these extreme values from distorting RMSE in euro-space.
        """
    )

st.divider()

# ═══════════════════════════════════════════════════════════════
# Section 3: Interactive Prediction Tool
# ═══════════════════════════════════════════════════════════════
st.subheader("🎮 Interactive Prediction Tool")
st.markdown("*Adjust the sliders to estimate a player's market value:*")

col1, col2, col3 = st.columns(3)

with col1:
    pred_position = st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
    pred_age = st.slider("Age", 16, 40, 25)
    pred_foot = st.selectbox("Preferred Foot", ["right", "left", "both"])
    pred_league = st.selectbox("League Tier", ["Top 5 League", "Other League"])

with col2:
    pred_goals = st.slider("Goals This Season", 0, 50, 5)
    pred_assists = st.slider("Assists This Season", 0, 30, 3)
    pred_minutes = st.slider("Minutes Played", 450, 4500, 2000, step=90)
    pred_appearances = st.slider("Appearances", 5, 60, 25)

with col3:
    pred_cl = st.checkbox("Champions League Player?", value=False)
    pred_height = st.slider("Height (cm)", 160, 205, 180)
    pred_stadium = st.slider("Club Stadium Seats", 5000, 100000, 30000, step=5000)
    pred_confederation = st.selectbox("Confederation", ["UEFA", "CONMEBOL/CONCACAF", "AFC", "CAF", "Other"])

st.markdown("##### Prior Valuation & Transfer History")
st.caption("The previous season's market value is the single most predictive feature — "
           "it tells the model the player's established reputation. Set to €0 for unknown/youth players.")
pcol1, pcol2, pcol3 = st.columns(3)
with pcol1:
    prev_value_options = {
        "Unknown / Youth Player": 0,
        "€500K (Lower league regular)": 500_000,
        "€2M (Mid-table starter)": 2_000_000,
        "€5M (Good domestic league player)": 5_000_000,
        "€10M (Top league squad player)": 10_000_000,
        "€20M (Established top-league starter)": 20_000_000,
        "€40M (National team regular)": 40_000_000,
        "€60M (Elite — top scorer/playmaker)": 60_000_000,
        "€80M (World-class)": 80_000_000,
        "€100M+ (Ballon d'Or contender)": 100_000_000,
    }
    prev_value_label = st.selectbox("Previous Season Market Value", list(prev_value_options.keys()), index=0)
    pred_prev_value = prev_value_options[prev_value_label]
with pcol2:
    pred_num_transfers = st.slider("Number of Career Transfers", 0, 10, 2)
with pcol3:
    prev_fee_options = {
        "€0 (Free / Academy)": 0,
        "€1M": 1_000_000,
        "€5M": 5_000_000,
        "€15M": 15_000_000,
        "€30M": 30_000_000,
        "€50M": 50_000_000,
        "€80M+": 80_000_000,
    }
    prev_fee_label = st.selectbox("Highest Previous Transfer Fee", list(prev_fee_options.keys()), index=0)
    pred_highest_fee = prev_fee_options[prev_fee_label]

if st.button("⚽ Predict Market Value", type="primary", use_container_width=True):
    # Build feature vector
    minutes_safe = max(pred_minutes, 1)
    goal_inv = pred_goals + pred_assists
    input_data = pd.DataFrame([{
        "age": pred_age,
        "age_squared": pred_age ** 2,
        "is_peak_age": 1 if 25 <= pred_age <= 29 else 0,
        "height_in_cm": pred_height,
        "stadium_seats": pred_stadium,
        "league_tier": 1 if pred_league == "Top 5 League" else 0,
        "champions_league_flag": 1 if pred_cl else 0,
        "champions_league_apps": 6 if pred_cl else 0,
        "total_goals": pred_goals,
        "total_assists": pred_assists,
        "total_minutes": pred_minutes,
        "num_appearances": pred_appearances,
        "total_yellow_cards": 3,
        "total_red_cards": 0,
        "goals_per_90": (pred_goals / minutes_safe) * 90,
        "assists_per_90": (pred_assists / minutes_safe) * 90,
        "yellow_cards_per_90": (3 / minutes_safe) * 90,
        "red_cards_per_90": 0,
        "goal_involvement": goal_inv,
        "minutes_per_goal_involvement": minutes_safe / goal_inv if goal_inv > 0 else 9999,
        "goal_involvement_per_app": goal_inv / pred_appearances if pred_appearances > 0 else 0,
        "goal_involvement_x_league": goal_inv * (1 if pred_league == "Top 5 League" else 0),
        "num_transfers": pred_num_transfers,
        "highest_previous_fee": pred_highest_fee,
        "total_transfer_fees": pred_highest_fee,
        "log_prev_season_value": np.log1p(pred_prev_value),
        "position_group": pred_position,
        "foot": pred_foot,
        "confederation": pred_confederation,
        # Age x position interactions
        "age_x_attack": pred_age if pred_position == "Attack" else 0,
        "age_x_midfield": pred_age if pred_position == "Midfield" else 0,
        "age_x_defender": pred_age if pred_position == "Defender" else 0,
        "age_x_goalkeeper": pred_age if pred_position == "Goalkeeper" else 0,
    }])

    # Predict with multiple models
    st.markdown("### 💰 Estimated Market Values")
    pred_cols_display = st.columns(min(4, len(results)))

    model_files = {
        "XGBoost": "xgboost",
        "Random Forest": "random_forest",
        "Decision Tree": "decision_tree",
        "Linear Regression": "linear_regression",
        "Ridge Regression": "ridge_regression",
        "KNN Regressor": "knn_regressor",
        "Lasso Regression": "lasso_regression",
    }

    predictions_live = {}
    for i, (display_name, file_name) in enumerate(model_files.items()):
        model = load_model(file_name)
        if model is not None:
            try:
                log_pred = model.predict(input_data)[0]
                log_pred = max(log_pred, 0)  # clip negative
                eur_pred = np.expm1(log_pred)
                predictions_live[display_name] = eur_pred

                # Get error margin from model MAE
                mae = model_mae.get(display_name, 0)

                with pred_cols_display[i % len(pred_cols_display)]:
                    if eur_pred >= 1e6:
                        display_val = f"€{eur_pred/1e6:.1f}M"
                        error_display = f"± €{mae/1e6:.1f}M"
                    else:
                        display_val = f"€{eur_pred:,.0f}"
                        error_display = f"± €{mae:,.0f}"
                    st.metric(display_name, display_val)
                    st.caption(f"Error margin: {error_display}")
            except Exception as e:
                with pred_cols_display[i % len(pred_cols_display)]:
                    st.metric(display_name, "Error")
                    st.caption(f"Prediction failed: {str(e)[:50]}")

    # Highlight best model
    if predictions_live:
        # Use XGBoost as the "best" (highest R²) rather than highest value
        best_model = "XGBoost" if "XGBoost" in predictions_live else max(predictions_live, key=predictions_live.get)
        best_val = predictions_live[best_model]
        best_mae = model_mae.get(best_model, 0)
        st.success(
            f"**Best model ({best_model}) estimates: €{best_val/1e6:.1f}M** "
            f"(expected error ± €{best_mae/1e6:.1f}M)"
        )

st.divider()

# ═══════════════════════════════════════════════════════════════
# Section 4: Undervalued & Overvalued Players (Latest Season)
# ═══════════════════════════════════════════════════════════════
st.subheader("💎 Most Undervalued & Overvalued Players (Latest Season)")
st.markdown("*Players from the latest season where XGBoost predictions diverge most from actual market value.*")

# Use XGBoost predictions
xgb_col = [c for c in predictions.columns if "xgboost" in c]
if xgb_col:
    xgb_col = xgb_col[0]
    undervalued = predictions.copy()
    undervalued["actual_eur"] = np.expm1(undervalued["y_true_log"])
    undervalued["predicted_eur"] = np.expm1(undervalued[xgb_col])
    undervalued["diff_eur"] = undervalued["predicted_eur"] - undervalued["actual_eur"]
    undervalued["diff_pct"] = (
        undervalued["diff_eur"] / undervalued["actual_eur"] * 100
    )

    # Filter to latest season only
    if "season" in undervalued.columns:
        latest_season = undervalued["season"].max()
        undervalued = undervalued[undervalued["season"] == latest_season]
        st.info(f"Showing results for season **{int(latest_season)}/{int(latest_season)+1}** only")

    # Filter to meaningful players (actual > 1M)
    undervalued = undervalued[undervalued["actual_eur"] > 1_000_000]

    def format_table(df_sub, ascending=True):
        display_df = df_sub[["player_name", "position_group", "club_name",
                             "actual_eur", "predicted_eur", "diff_pct"]].copy()
        display_df["Actual Value"] = display_df["actual_eur"].apply(lambda x: f"€{x:,.0f}")
        display_df["Predicted Value"] = display_df["predicted_eur"].apply(lambda x: f"€{x:,.0f}")
        display_df["Difference"] = display_df["diff_pct"].apply(lambda x: f"{x:+.0f}%")
        display_df = display_df.rename(columns={
            "player_name": "Player", "position_group": "Position", "club_name": "Club",
        })
        return display_df[["Player", "Position", "Club", "Actual Value", "Predicted Value", "Difference"]]

    tab_under, tab_over = st.tabs(["🟢 Most Undervalued", "🔴 Most Overvalued"])

    with tab_under:
        bargains = undervalued.nlargest(15, "diff_pct")
        if len(bargains) > 0:
            st.dataframe(format_table(bargains), hide_index=True, use_container_width=True)
            st.caption("*Players whose on-pitch performance suggests they are undervalued by the market — potential transfer bargains.*")
        else:
            st.info("No significantly undervalued players found.")

    with tab_over:
        overpriced = undervalued.nsmallest(15, "diff_pct")
        if len(overpriced) > 0:
            st.dataframe(format_table(overpriced, ascending=False), hide_index=True, use_container_width=True)
            st.caption("*Players whose market value may exceed what their performance justifies — potential overpays.*")
        else:
            st.info("No significantly overvalued players found.")

st.markdown(
    """
    > **Note on accuracy:** Market value is influenced by factors outside our data — agent
    > negotiations, social media presence, marketability, injury history, and club finances.
    > Our XGBoost model (R² = 0.92) captures the **explainable portion** of market value
    > based on on-pitch performance, league context, and prior valuation history.
    """
)
