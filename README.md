---
title: Football Player Market Value Predictor
emoji: "\u26BD"
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
---

# Football Player Market Value Predictor

A machine learning web application that predicts football (soccer) player market values using the Transfermarkt dataset. Built as a university group project for a Data Management and Analysis course.

**Best model: XGBoost — R² = 0.9206, MAE = €1,378,285**

---

## What It Does

Football clubs spend billions on transfers yearly, often overpaying or missing undervalued talent. This tool estimates a player's market value based on:

- On-pitch performance (goals, assists, minutes, per-90 stats)
- Player profile (age, position, foot, nationality)
- Club and league context (top-5 league flag, stadium size, Champions League)
- Transfer history (number of transfers, fees paid)
- Previous season's valuation (the single most predictive feature)

The app has five interactive pages: business case overview, data visualisations, live predictions, SHAP explainability, and a hyperparameter tuning analysis.

---

## Project Structure

```
pods_final/
├── app.py                          # Streamlit entry point
├── run_pipeline.py                 # Runs data pipeline end-to-end
├── benchmark_quick.py              # Quick XGBoost benchmark
├── requirements.txt
│
├── src/
│   ├── data_loader.py              # Load + join 7 CSV tables
│   ├── feature_engineering.py     # All feature creation logic
│   ├── models.py                   # Train, evaluate, and save all models
│   └── shap_analysis.py            # Pre-compute SHAP values
│
├── pages/
│   ├── 1_Business_Case.py
│   ├── 2_Visualizations.py        # 11 charts + player lookup with prediction
│   ├── 3_Predictions.py           # Model comparison + interactive predictor
│   ├── 4_Explainability.py        # SHAP analysis
│   └── 5_Hyperparameter_Tuning.py
│
├── data/
│   ├── *.csv                       # Raw Transfermarkt CSVs (from Kaggle)
│   └── processed/
│       └── features.parquet        # Engineered feature matrix
│
└── models/
    ├── xgboost.pkl                 # Best model
    ├── random_forest.pkl
    ├── decision_tree.pkl
    ├── linear_regression.pkl
    ├── ridge_regression.pkl
    ├── lasso_regression.pkl
    ├── knn_regressor.pkl
    ├── model_results.csv           # Metrics for all models
    ├── test_predictions.parquet    # Test set predictions
    ├── shap_values.pkl             # Pre-computed SHAP values
    ├── shap_by_position.pkl        # Position-specific SHAP
    └── feature_info.json           # Feature list metadata
```

---

## Dataset

**Source:** [Transfermarkt dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)

| Stat | Value |
|---|---|
| Player-season records | 91,823 |
| Unique players | 26,644 |
| Seasons covered | 2012 – 2025 |
| Positions | Defender (34%), Midfielder (30%), Attacker (28%), Goalkeeper (9%) |

The dataset consists of 10 relational CSV tables: `players`, `appearances`, `player_valuations`, `games`, `competitions`, `clubs`, `transfers`, `club_games`, `game_events`, `game_lineups`.

---

## Model Performance

All models predict `log1p(market_value_in_eur)` (log-transformed for normality), then inverse-transform predictions for display.

| Model | R² | CV R² | MAE |
|---|---|---|---|
| **XGBoost** | **0.9206** | 0.9212 ± 0.0014 | €1,378,285 |
| Random Forest | 0.9094 | 0.9109 ± 0.0027 | €1,458,831 |
| Decision Tree | 0.8828 | 0.8846 ± 0.0019 | €1,681,739 |
| KNN Regressor | 0.7580 | 0.7522 ± 0.0046 | €2,565,391 |
| Linear Regression | 0.7578 | 0.7562 ± 0.0045 | €2,867,637 |
| Ridge Regression | 0.7578 | 0.7562 ± 0.0045 | €2,867,494 |
| Lasso Regression | 0.7247 | 0.7238 ± 0.0045 | €2,909,894 |

### Feature Engineering Journey

The biggest performance gains came from features, not hyperparameter tuning:

| Step | R² | Delta |
|---|---|---|
| Baseline (age, goals, assists, league, club, transfers) | 0.8158 | — |
| + Previous season market value (lag feature) | 0.9200 | **+0.1041** |
| + Goal involvement per appearance | 0.9200 | +0.0008 |
| + Age × position interactions | 0.9200 | +0.0001 |
| + Goal involvement × league tier | **0.9206** | +0.0006 |

The previous season's valuation alone accounts for a **+0.1041 R² improvement** — more than any hyperparameter change could deliver.

---

## Features Used

**Numeric (30 features):**

| Category | Features |
|---|---|
| Player profile | age, age_squared, is_peak_age, height_in_cm |
| Age × position | age_x_attack, age_x_midfield, age_x_defender, age_x_goalkeeper |
| Club/league | stadium_seats, league_tier, champions_league_flag, champions_league_apps |
| Raw performance | total_goals, total_assists, total_minutes, num_appearances, total_yellow_cards, total_red_cards |
| Per-90 stats | goals_per_90, assists_per_90, yellow_cards_per_90, red_cards_per_90 |
| Derived | goal_involvement, minutes_per_goal_involvement, goal_involvement_per_app, goal_involvement_x_league |
| Transfer history | num_transfers, highest_previous_fee, total_transfer_fees |
| Lag feature | log_prev_season_value |

**Categorical (3 features, one-hot encoded):** position_group, foot, confederation

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- The Transfermarkt CSV files placed in `data/` (download from Kaggle link above)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the data pipeline

This loads the raw CSVs, joins them, engineers features, and saves `data/processed/features.parquet`.

```bash
python run_pipeline.py
```

### Train all models

Trains all 7 models, saves `.pkl` files, and writes `models/model_results.csv`.

```bash
python src/models.py
```

### Pre-compute SHAP values

```bash
python -m src.shap_analysis
```

### Launch the app

```bash
streamlit run app.py
```

### Quick benchmark (XGBoost only, for testing)

```bash
python benchmark_quick.py
```

---

## App Pages

### 1. Business Case
Overview of the problem, dataset statistics, entity-relationship diagram, and fun facts (most valuable player, biggest transfer, etc.).

### 2. Visualisations
11 interactive charts with sidebar filters (position, age range, season):
- Market value distribution (raw vs log-transformed)
- Value by position (violin plot)
- The age curve (peak at ~27)
- League premium (top 5 vs rest)
- Foot preference analysis
- Champions League premium
- Goals/assists vs value scatter
- Feature correlation heatmap
- Top 15 clubs by squad value
- Market value inflation over time
- Value by confederation
- **Player Lookup** — search any player to see their value trajectory + predicted next-season value

### 3. Predictions
- Model comparison table and bar chart
- Predicted vs actual scatter plot
- Interactive prediction tool (select age, goals, assists, league, etc.)
- "Undervalued players" finder

### 4. Explainability (SHAP)
- Global beeswarm plot with human-readable feature names
- Mean absolute SHAP bar chart
- Position-specific feature importance (what drives value for strikers vs defenders)
- Individual player SHAP waterfall breakdown
- Dependence plots for top 4 features

### 5. Hyperparameter Tuning
- Feature engineering impact (ablation study waterfall chart)
- Current model performance table (live from `model_results.csv`)
- Interactive tuning demos: KNN k, Ridge/Lasso alpha, Decision Tree depth
- Comparison: feature engineering (+0.1048 R²) vs hyperparameter tuning (+~0.009 R²)
- Key takeaways and W&B methodology

---

## Tech Stack

| Component | Library |
|---|---|
| Web app | Streamlit |
| Data processing | Pandas, NumPy |
| ML models | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Model serialisation | joblib |
| Data storage | Parquet (via PyArrow) |
| Experiment tracking | Weights & Biases (wandb) |
