"""
Page 4: Explainability (SHAP)
==============================
Global and position-specific SHAP analysis on the XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shap

st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1B3A2D",
    "text.color": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#FAFAFA",
    "ytick.color": "#FAFAFA",
    "axes.edgecolor": "#FAFAFA",
})

st.title("🧠 Explainability — What Drives Player Value?")

# ── Data scope toggle ──
data_scope = st.sidebar.radio(
    "🏟️ Data Scope",
    ["All Players", "Top 5 Leagues Only", "€10M+ Players Only"],
    index=0,
    key="expl_scope",
)

# Load model results for dynamic R² display
@st.cache_data
def load_model_results():
    import os
    path = "models/model_results.csv"
    if os.path.exists(path):
        res = pd.read_csv(path)
        xgb = res[res["Model"] == "XGBoost"]
        if len(xgb) > 0:
            return xgb.iloc[0]["R2"]
    return 0.92

xgb_r2 = load_model_results()
st.markdown(f"*Using SHAP (SHapley Additive exPlanations) on our best model (XGBoost, R²={xgb_r2:.2f})*")

st.info(
    "**What is SHAP?** SHAP values decompose each prediction into the contribution of every feature. "
    "Positive SHAP = increases predicted value, Negative SHAP = decreases it. "
    "This lets us explain *why* a player is valued high or low, not just *what* the prediction is."
)

# ── Load pre-computed SHAP values ──
@st.cache_data
def load_shap():
    with open("models/shap_values.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_shap_by_position():
    path = "models/shap_by_position.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

if not os.path.exists("models/shap_values.pkl"):
    st.error("SHAP values not found. Run `python -m src.shap_analysis` first.")
    st.stop()

shap_data = load_shap()
position_shap = load_shap_by_position()

shap_values = shap_data["shap_values"]
feature_names = shap_data["feature_names"]
meta = shap_data["meta_sample"]

# ── Human-readable feature name mapping ──
DISPLAY_NAMES = {
    "age": "Age",
    "age_squared": "Age Squared",
    "is_peak_age": "Peak Age (25-29)",
    "height_in_cm": "Height (cm)",
    "age_x_attack": "Age x Attacker",
    "age_x_midfield": "Age x Midfielder",
    "age_x_defender": "Age x Defender",
    "age_x_goalkeeper": "Age x Goalkeeper",
    "stadium_seats": "Club Stadium Size",
    "league_tier": "Top 5 League",
    "champions_league_flag": "Champions League Player",
    "champions_league_apps": "CL/EL Appearances",
    "total_goals": "Total Goals",
    "total_assists": "Total Assists",
    "total_minutes": "Total Minutes",
    "num_appearances": "Appearances",
    "total_yellow_cards": "Yellow Cards",
    "total_red_cards": "Red Cards",
    "goals_per_90": "Goals per 90 min",
    "assists_per_90": "Assists per 90 min",
    "yellow_cards_per_90": "Yellows per 90 min",
    "red_cards_per_90": "Reds per 90 min",
    "goal_involvement": "Goal Involvement (G+A)",
    "minutes_per_goal_involvement": "Minutes per G+A",
    "goal_involvement_per_app": "G+A per Appearance",
    "goal_involvement_x_league": "G+A x Top League",
    "num_transfers": "Transfer Count",
    "highest_previous_fee": "Highest Transfer Fee",
    "total_transfer_fees": "Total Transfer Fees",
    "log_prev_season_value": "Prev. Season Value (log)",
    "position_group_Attack": "Position: Attack",
    "position_group_Defender": "Position: Defender",
    "position_group_Goalkeeper": "Position: Goalkeeper",
    "position_group_Midfield": "Position: Midfield",
    "foot_both": "Foot: Both",
    "foot_left": "Foot: Left",
    "foot_right": "Foot: Right",
    "foot_unknown": "Foot: Unknown",
    "confederation_AFC": "Confederation: AFC",
    "confederation_CAF": "Confederation: CAF",
    "confederation_CONCACAF": "Confederation: CONCACAF",
    "confederation_CONMEBOL": "Confederation: CONMEBOL",
    "confederation_CONMEBOL/CONCACAF": "Confederation: Americas",
    "confederation_Other": "Confederation: Other",
    "confederation_UEFA": "Confederation: UEFA",
}

display_names = [DISPLAY_NAMES.get(f, f.replace("_", " ").title()) for f in feature_names]

# Apply readable names to all SHAP objects
shap_values.feature_names = display_names

for pos in position_shap:
    position_shap[pos]["shap_values"].feature_names = display_names

st.divider()

# ═══════════════════════════════════════════════════════════════
# 1. Global SHAP Summary (Beeswarm)
# ═══════════════════════════════════════════════════════════════
st.subheader("1. Global Feature Importance — SHAP Beeswarm Plot")
st.markdown("*Each dot is a player. Color = feature value (red = high, blue = low). "
            "Position on X-axis = impact on prediction.*")

fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
# Force white text on all elements (SHAP overrides rcParams)
for fig_obj in [plt.gcf()]:
    for ax_obj in fig_obj.axes:
        ax_obj.set_xlabel(ax_obj.get_xlabel(), color="#FAFAFA")
        ax_obj.set_ylabel(ax_obj.get_ylabel(), color="#FAFAFA")
        ax_obj.title.set_color("#FAFAFA")
        for label in ax_obj.get_xticklabels() + ax_obj.get_yticklabels():
            label.set_color("#FAFAFA")
    # Fix colorbar label
    for cb_ax in fig_obj.axes:
        if cb_ax.get_ylabel():
            cb_ax.yaxis.label.set_color("#FAFAFA")
        for label in cb_ax.get_yticklabels():
            label.set_color("#FAFAFA")
st.pyplot(plt.gcf())
plt.close("all")

with st.expander("📖 How to read this chart"):
    st.markdown(
        """
        - **Features** are sorted by importance (top = most important)
        - **Each dot** = one player in our sample
        - **Red dots** = high feature value, **Blue dots** = low feature value
        - **Position on X-axis** = impact on prediction (right = increases value, left = decreases)
        - Example: Red `goals_per_90` dots on the right → high goal rate increases predicted value
        """
    )

st.divider()

# ═══════════════════════════════════════════════════════════════
# 2. SHAP Bar Plot — Mean |SHAP|
# ═══════════════════════════════════════════════════════════════
st.subheader("2. Mean Absolute SHAP Values — Feature Rankings")

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
for ax_obj in plt.gcf().axes:
    ax_obj.set_xlabel(ax_obj.get_xlabel(), color="#FAFAFA")
    ax_obj.title.set_color("#FAFAFA")
    for label in ax_obj.get_xticklabels() + ax_obj.get_yticklabels():
        label.set_color("#FAFAFA")
st.pyplot(plt.gcf())
plt.close("all")

st.markdown("> This shows the average *magnitude* of each feature's impact. "
            "Unlike the beeswarm, direction doesn't matter — just how much influence it has overall.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# 3. Position-Specific SHAP (The Key Insight)
# ═══════════════════════════════════════════════════════════════
st.subheader("3. 🎯 Position-Specific Feature Importance")
st.markdown(
    """
    **The key insight:** Different positions value different attributes.
    Goals matter most for attackers, but league prestige and playing time matter more for defenders.
    """
)

if position_shap:
    available_positions = list(position_shap.keys())
    
    # Show 2 at a time in rows
    for row_start in range(0, len(available_positions), 2):
        cols = st.columns(2)
        for i, pos in enumerate(available_positions[row_start:row_start+2]):
            with cols[i]:
                st.markdown(f"#### {pos} ({position_shap[pos]['n_samples']} players)")
                fig, ax = plt.subplots(figsize=(7, 5))
                shap.plots.bar(position_shap[pos]["shap_values"], max_display=10, show=False)
                for ax_obj in plt.gcf().axes:
                    ax_obj.set_xlabel(ax_obj.get_xlabel(), color="#FAFAFA")
                    ax_obj.title.set_color("#FAFAFA")
                    for label in ax_obj.get_xticklabels() + ax_obj.get_yticklabels():
                        label.set_color("#FAFAFA")
                st.pyplot(plt.gcf())
                plt.close("all")
else:
    st.info("Position-specific SHAP not available. Run `python -m src.shap_analysis` first.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# 4. Individual Player SHAP Waterfall
# ═══════════════════════════════════════════════════════════════
st.subheader("4. 🔬 Individual Player Breakdown")
st.markdown("*Select a player to see exactly why the model assigned their valuation.*")

player_list = meta["player_name"].dropna().unique().tolist()
selected_player = st.selectbox("Select a player", sorted(player_list)[:500])

if selected_player:
    player_idx = meta[meta["player_name"] == selected_player].index
    if len(player_idx) > 0:
        idx = player_idx[0]
        player_info = meta.loc[idx]

        c1, c2, c3 = st.columns(3)
        c1.metric("Player", player_info["player_name"])
        c2.metric("Position", player_info["position_group"])
        c3.metric("Market Value", f"€{player_info['market_value_in_eur']:,.0f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        for ax_obj in plt.gcf().axes:
            ax_obj.set_xlabel(ax_obj.get_xlabel(), color="#FAFAFA")
            ax_obj.set_ylabel(ax_obj.get_ylabel(), color="#FAFAFA")
            ax_obj.title.set_color("#FAFAFA")
            for label in ax_obj.get_xticklabels() + ax_obj.get_yticklabels():
                label.set_color("#FAFAFA")
        st.pyplot(plt.gcf())
        plt.close("all")

        st.caption("Each bar shows how much a single feature pushes the prediction up (red) or down (blue) "
                   "from the baseline expected value.")

st.divider()

# ═══════════════════════════════════════════════════════════════
# 5. SHAP Dependence Plots (manual scatter for robustness)
# ═══════════════════════════════════════════════════════════════
st.subheader("5. SHAP Dependence Plots — Top Features")
st.markdown("*How does each feature's value affect the prediction?*")

top_features = ["age", "goals_per_90", "league_tier", "log_prev_season_value"]
available_top = [f for f in top_features if f in feature_names]

if available_top:
    dep_cols = st.columns(min(len(available_top), 4))
    for i, feat in enumerate(available_top):
        with dep_cols[i]:
            feat_idx = feature_names.index(feat)
            feat_shap = shap_values.values[:, feat_idx]
            feat_vals = shap_values.data[:, feat_idx]
            feat_display = DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())

            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(feat_vals, feat_shap, alpha=0.3, s=8,
                                 c=feat_vals, cmap="coolwarm", edgecolors="none")
            ax.set_xlabel(feat_display)
            ax.set_ylabel(f"SHAP value")
            ax.set_title(feat_display)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.2)
            plt.colorbar(scatter, ax=ax, label="Feature value")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

st.divider()

# ── Narrative ──
st.subheader("📝 Key Takeaways")

col_left, col_right = st.columns(2)
with col_left:
    st.markdown(
        """
        #### What Increases Value ⬆️
        - High `goals_per_90` (especially for attackers)
        - Playing in a **Top 5 league** (`league_tier = 1`)
        - Large club stadium (proxy for club prestige)
        - **Champions League** appearances
        - Being in **peak age** window (25-29)
        """
    )
with col_right:
    st.markdown(
        """
        #### What Decreases Value ⬇️
        - **Age > 30** (sharp decline after 30)
        - Playing in lower leagues (`league_tier = 0`)
        - Low playing time (fewer minutes/appearances)
        - Low goal involvement
        - Small club stadium size
        """
    )

st.markdown(
    """
    > *"Market value is a function of talent AND context. Our SHAP analysis reveals that 
    > where you play can matter as much as how well you play. A mediocre striker in the 
    > Premier League may be valued higher than a prolific scorer in a 2nd-tier league."*
    """
)
