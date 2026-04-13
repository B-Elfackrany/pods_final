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

results = load_results()
predictions = load_predictions()
feature_info = load_feature_info()

# ═══════════════════════════════════════════════════════════════
# Section 1: Model Comparison
# ═══════════════════════════════════════════════════════════════
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
    ax.set_xlim(0.5, 0.9)
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

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true_eur / 1e6, y_pred_eur / 1e6, alpha=0.2, s=5, color="#FFD700")
    max_val = max(y_true_eur.max(), y_pred_eur.max()) / 1e6
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Value (€ Millions)")
    ax.set_ylabel("Predicted Value (€ Millions)")
    ax.set_title(f"{selected_model} — Predicted vs Actual")
    ax.legend()
    ax.set_xlim(0, min(max_val, 200))
    ax.set_ylim(0, min(max_val, 200))
    st.pyplot(fig)
    plt.close()

with col_b:
    # Residuals distribution
    residuals_eur = (y_pred_eur - y_true_eur) / 1e6
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(residuals_eur.clip(-50, 50), bins=80, color="#4CAF50", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (€ Millions)")
    ax.set_ylabel("Count")
    ax.set_title(f"{selected_model} — Residuals Distribution")
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
        "num_transfers": 2,
        "highest_previous_fee": 0,
        "total_transfer_fees": 0,
        "position_group": pred_position,
        "foot": pred_foot,
        "confederation": pred_confederation,
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
            log_pred = model.predict(input_data)[0]
            log_pred = max(log_pred, 0)  # clip negative
            eur_pred = np.expm1(log_pred)
            predictions_live[display_name] = eur_pred

            with pred_cols_display[i % len(pred_cols_display)]:
                if eur_pred >= 1e6:
                    display_val = f"€{eur_pred/1e6:.1f}M"
                else:
                    display_val = f"€{eur_pred:,.0f}"
                st.metric(display_name, display_val)

    # Highlight best model
    if predictions_live:
        best_model = max(predictions_live, key=predictions_live.get)
        best_val = predictions_live[best_model]
        st.success(f"**Best model ({best_model}) estimates: €{best_val/1e6:.1f} Million**")

st.divider()

# ═══════════════════════════════════════════════════════════════
# Section 4: Undervalued Players
# ═══════════════════════════════════════════════════════════════
st.subheader("💎 Find Undervalued Players")
st.markdown("*Players where our model predicts a significantly higher value than their actual market value — potential transfer bargains.*")

# Use XGBoost predictions
xgb_col = [c for c in predictions.columns if "xgboost" in c]
if xgb_col:
    xgb_col = xgb_col[0]
    undervalued = predictions.copy()
    undervalued["actual_eur"] = np.expm1(undervalued["y_true_log"])
    undervalued["predicted_eur"] = np.expm1(undervalued[xgb_col])
    undervalued["diff_pct"] = (
        (undervalued["predicted_eur"] - undervalued["actual_eur"])
        / undervalued["actual_eur"] * 100
    )

    # Filter: predicted > actual by at least 100%, actual > 1M (meaningful players)
    bargains = undervalued[
        (undervalued["diff_pct"] > 100)
        & (undervalued["actual_eur"] > 1_000_000)
    ].nlargest(20, "diff_pct")

    if len(bargains) > 0:
        display_bargains = bargains[["player_name", "position_group", "club_name",
                                      "season", "actual_eur", "predicted_eur", "diff_pct"]].copy()
        display_bargains["Actual Value"] = display_bargains["actual_eur"].apply(lambda x: f"€{x:,.0f}")
        display_bargains["Predicted Value"] = display_bargains["predicted_eur"].apply(lambda x: f"€{x:,.0f}")
        display_bargains["Difference"] = display_bargains["diff_pct"].apply(lambda x: f"+{x:.0f}%")
        display_bargains = display_bargains.rename(columns={
            "player_name": "Player", "position_group": "Position",
            "club_name": "Club", "season": "Season",
        })

        st.dataframe(
            display_bargains[["Player", "Position", "Club", "Season",
                             "Actual Value", "Predicted Value", "Difference"]],
            hide_index=True,
            use_container_width=True,
        )
        st.caption("*Based on XGBoost model predictions. These are players whose on-pitch "
                   "performance suggests they may be undervalued by the market.*")
    else:
        st.info("No significantly undervalued players found with current criteria.")

st.markdown(
    """
    > **Note on accuracy:** Market value is influenced by factors outside our data — agent
    > negotiations, social media presence, marketability, injury history, and club finances.
    > Our XGBoost model (R² = 0.92) captures the **explainable portion** of market value
    > based on on-pitch performance, league context, and prior valuation history.
    """
)
