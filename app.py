"""
Football Player Market Value Predictor
=======================================
Main Streamlit entry point (multipage app).
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Football Player Market Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ──
with st.sidebar:
    st.title("⚽ Value Predictor")
    st.caption("Data-driven football player market value estimation")
    st.divider()

    @st.cache_data
    def get_quick_stats():
        df = pd.read_parquet("data/processed/features.parquet")
        results = pd.read_csv("models/model_results.csv")
        return {
            "players": f"{df['player_id'].nunique():,}",
            "records": f"{len(df):,}",
            "seasons": f"{int(df['season'].min())}–{int(df['season'].max())}",
            "models": str(len(results)),
        }
    try:
        stats = get_quick_stats()
        st.markdown(
            f"""
            **📊 Dataset**
            {stats['players']} players | {stats['records']} records
            Seasons {stats['seasons']} | {stats['models']} ML models trained
            """
        )
    except Exception:
        st.markdown("**Transfermarkt Dataset** — Player market values")

    st.divider()
    st.markdown(
        "*University Project*  \n"
        "*Data Management & Analysis*"
    )

# ── Home page content ──
st.title("⚽ Football Player Market Value Predictor")
st.markdown("### Moneyball for Football — A Data-Driven Approach")

st.markdown(
    """
    > Football clubs spend billions on transfers yearly, often overpaying or missing
    > undervalued talent. This tool estimates player market values using machine learning,
    > achieving **R² = 0.92** with XGBoost — explaining 92% of the variance in player value
    > using on-pitch performance, valuation history, and contextual data.
    """
)

st.divider()

# Hero metrics
try:
    stats = get_quick_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players", stats["players"])
    m2.metric("Player-Season Records", stats["records"])
    m3.metric("Seasons Covered", stats["seasons"])
    m4.metric("ML Models Trained", stats["models"])
except Exception:
    pass

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 **Explore** the data through 11 interactive visualizations with filters")
with col2:
    st.info("🔮 **Predict** any player's value with multiple ML models and find undervalued talent")
with col3:
    st.info("🧠 **Understand** what drives value with SHAP explainability analysis")

st.markdown(
    """
    #### Navigate the pages

    | Page | What's Inside |
    |---|---|
    | **Business Case** | Problem context, dataset structure, entity-relationship diagram, data quality |
    | **Visualizations** | 11 interactive charts — age curves, league premiums, CL effect, player lookup |
    | **Predictions** | Model comparison, live prediction tool, undervalued and overvalued players |
    | **Explainability** | SHAP analysis — global, by position, individual player breakdowns |
    | **Hyperparameter Tuning** | Feature engineering impact, tuning demos, W&B sweep results |
    """
)

st.divider()

st.markdown(
    """
    **Data source:** [Transfermarkt Football Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores) on Kaggle
    **Tech stack:** Streamlit, scikit-learn, XGBoost, SHAP, Plotly, Pandas
    **Best model:** XGBoost — R² = 0.92, MAE = ~€1.4M
    """
)
