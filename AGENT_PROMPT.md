# AI Agent Prompt — Football Player Market Value Predictor

You are building a Streamlit multipage app that predicts football (soccer) player market values using the Transfermarkt dataset from Kaggle. This is a university group project for a Data Management and Analysis course. The app must be polished, interview-worthy, and go beyond basic expectations.

---

## PROJECT OVERVIEW

**Business Problem:** Football clubs spend billions on transfers yearly, often overpaying or missing undervalued talent. Build a data-driven tool that estimates a player's market value based on performance stats, player profile, and contextual factors — essentially "Moneyball for football."

**Target Variable:** `market_value_in_eur` from the `player_valuations` table (continuous, use `np.log1p()` transformation since values are heavily right-skewed, and `np.expm1()` to inverse-transform predictions for display).

**Dataset:** Transfermarkt Kaggle dataset (https://www.kaggle.com/datasets/davidcariboo/player-scores). It contains 10 relational CSV tables: `competitions`, `games`, `clubs`, `players`, `appearances`, `player_valuations`, `club_games`, `game_events`, `game_lineups`, `transfers`. Download and place them in `data/raw/`.

---

## PROJECT STRUCTURE

```
football-value-predictor/
├── app.py                              # Main Streamlit entry point
├── pages/
│   ├── 1_🏠_Business_Case.py
│   ├── 2_📊_Visualizations.py
│   ├── 3_🔮_Predictions.py
│   ├── 4_🧠_Explainability.py
│   └── 5_⚙️_Hyperparameter_Tuning.py
├── src/
│   ├── data_loader.py                  # Load + join the 10 CSV tables
│   ├── feature_engineering.py          # All feature creation logic
│   ├── models.py                       # Train, evaluate, save all models
│   ├── shap_analysis.py                # SHAP computation helpers
│   └── wandb_tracking.py              # W&B sweep + logging utilities
├── data/
│   ├── raw/                            # Original CSVs from Kaggle
│   └── processed/                      # Engineered feature matrix (parquet)
├── models/                             # Saved trained models (.pkl via joblib)
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml                     # Theme config
```

---

## STEP 1: DATA PIPELINE (`src/data_loader.py`)

Load and join the following tables using pandas. Cache everything with `@st.cache_data` in Streamlit.

### Tables to load:
- `players.csv` → player_id, name, position, date_of_birth, height, foot, current_club_id, country_of_citizenship
- `appearances.csv` → player_id, game_id, goals, assists, minutes_played, yellow_cards, red_cards
- `player_valuations.csv` → player_id, date, market_value_in_eur, current_club_id
- `games.csv` → game_id, competition_id, season, date, home_club_id, away_club_id
- `competitions.csv` → competition_id, name, type, country_name
- `clubs.csv` → club_id, name, stadium_seats, domestic_competition_id
- `transfers.csv` → player_id, transfer_date, transfer_fee, from_club_id, to_club_id

### Join logic:
1. From `appearances`, aggregate per player per season:
   - total_goals, total_assists, total_minutes, total_yellow_cards, total_red_cards, num_appearances
   - Compute per-90 stats: goals_per_90 = (total_goals / total_minutes) * 90, same for assists, cards
2. Join `appearances` with `games` to get `competition_id` and `season`
3. Join with `competitions` to get league name and type
4. Count Champions League / Europa League appearances per player per season
5. Join with `players` for profile info (age at valuation date, position, foot, nationality)
6. Join with `clubs` for club prestige (use `stadium_seats` as proxy)
7. From `transfers`, aggregate per player: num_previous_transfers, highest_previous_fee, total_fees
8. Join with `player_valuations` to get the target variable, matching on player_id and closest season

### Output:
A single flat DataFrame saved as `data/processed/features.parquet` with columns like:
```
player_id, player_name, season, age, position, foot, nationality_confederation,
total_goals, total_assists, total_minutes, num_appearances,
goals_per_90, assists_per_90, yellow_cards_per_90, red_cards_per_90,
league_tier (top5 vs other), club_stadium_seats, champions_league_apps,
num_transfers, highest_previous_fee, years_at_club,
goal_involvement, minutes_per_goal_involvement, age_squared, is_peak_age,
market_value_in_eur (TARGET), log_market_value (log-transformed TARGET)
```

---

## STEP 2: FEATURE ENGINEERING (`src/feature_engineering.py`)

### Features to create:

**From appearances (per player per season):**
- total_goals, total_assists, total_minutes, num_appearances
- goals_per_90 = (total_goals / total_minutes) * 90 (handle division by zero)
- assists_per_90 = (total_assists / total_minutes) * 90
- yellow_cards_per_90, red_cards_per_90
- goal_involvement = total_goals + total_assists
- minutes_per_goal_involvement = total_minutes / goal_involvement (cap at max if 0 involvement)

**From players:**
- age = valuation_year - birth_year
- age_squared = age ** 2 (captures non-linear aging curve)
- is_peak_age = 1 if 25 <= age <= 29 else 0
- position_group: map sub_position to Attack / Midfield / Defense / Goalkeeper
- foot: one-hot encode (left / right / both)

**From clubs + competitions:**
- league_tier: 1 if competition is in [Premier League, La Liga, Bundesliga, Serie A, Ligue 1] else 0
- club_prestige: stadium_seats (or normalize it)
- champions_league_flag: 1 if player appeared in CL games that season

**From transfers:**
- num_previous_transfers: count of transfers before current valuation date
- highest_previous_fee: max transfer_fee before current date
- years_at_current_club: valuation_date - join_date

**Handle categoricals:**
- position_group → one-hot encode
- foot → one-hot encode
- nationality → group into confederations (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC) then one-hot
- For skrub pipeline: keep as strings — TableVectorizer handles it automatically

**Handle missing values:**
- For per-90 stats where minutes = 0: fill with 0
- For transfer fees where missing: fill with 0 (never been transferred)
- For height where missing: fill with median by position
- Drop rows where market_value_in_eur is null

---

## STEP 3: MODELS (`src/models.py`)

Train and evaluate these models. All should use scikit-learn compatible API (fit/predict).

### Model 1: Linear Regression (OLS) — Baseline
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

### Model 2: Ridge Regression
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # tune alpha via W&B
```

### Model 3: Lasso Regression
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)  # tune alpha; also shows automatic feature selection via zero coefficients
```

### Model 4: Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=10, min_samples_split=20)
```

### Model 5: KNN Regressor
```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=10, weights='distance')
```

### Model 6: Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
```

### Model 7: XGBoost Regressor
```python
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
```

### Model 8 (Extra Mile): skrub tabular_learner
```python
from skrub import tabular_learner
# Pass the RAW dataframe (with string columns) — skrub handles encoding automatically
model = tabular_learner('regressor')
```

### Model 9 (Extra Mile / Bonus): CARTE Regressor
```python
from carte_ai import CARTERegressor
model = CARTERegressor(num_model=10, device="cpu", random_state=42)
# Note: requires PyTorch, may be complex to install. Treat as optional bonus.
```

### Evaluation:
- Train/test split: 80/20, stratified by position_group
- Also run 5-fold cross-validation for robust metrics
- Metrics: R², MAE, RMSE, MAPE
- Save trained models with `joblib.dump()` to `models/` directory
- Always train on log-transformed target. Inverse-transform (`np.expm1`) for display metrics.

### Important preprocessing:
- For sklearn models (1-7): use sklearn Pipeline → StandardScaler for numeric + OneHotEncoder for categoricals
- For skrub (model 8): pass raw DataFrame with string columns — TableVectorizer handles it
- For CARTE (model 9): pass raw DataFrame with string columns
- Filter to players with >= 450 minutes (5 full games) to avoid noisy per-90 stats

---

## STEP 4: STREAMLIT PAGES

### Page 1: 🏠 Business Case & Data Presentation (`pages/1_🏠_Business_Case.py`)

- Title: "⚽ Football Player Market Value Predictor"
- Business problem explanation with context (transfer market size, examples of overpaying)
- Key stats in `st.metric` cards: total players, total appearances, leagues covered, date range
- Entity-relationship diagram showing how the 10 tables connect
- `st.dataframe` with sample of processed features
- Missing values summary + data quality notes
- "Fun facts" section: most valuable player, biggest transfer, etc.

### Page 2: 📊 Data Visualization (`pages/2_📊_Visualizations.py`)

Build 6-8 charts using Seaborn + Matplotlib (required by assignment), optionally Plotly for interactivity:

1. Market value distribution — histogram + log-transformed version side by side
2. Value by position — box plot (Attack > Midfield > Defense > GK)
3. The age curve — line plot of avg market value by age (peak at ~27)
4. League premium — bar chart of avg player value by top leagues
5. Goals/Assists vs Value — scatter plot with regression line
6. Correlation heatmap — all numeric features vs log_market_value
7. Top 15 clubs by total squad value — horizontal bar chart
8. Market value inflation over time — median value per year

Add interactivity:
- `st.selectbox` to filter by position group
- `st.slider` to filter by age range
- Player lookup: `st.text_input` to search a name and show their value trajectory over time

### Page 3: 🔮 Predictions (`pages/3_🔮_Predictions.py`)

**Model comparison section:**
- Table: Model Name | R² | MAE (€) | RMSE (€) | MAPE (%) | Training Time
- Bar chart comparing R² across all models
- Predicted vs Actual scatter plot with `st.selectbox` to toggle models
- Residuals distribution plot

**Interactive prediction tool:**
- `st.selectbox`: Position (Attack / Midfield / Defense / Goalkeeper)
- `st.slider`: Age (16-40), Goals (0-50), Assists (0-30), Minutes (0-4000), Appearances (0-60)
- `st.selectbox`: League tier (Top 5 / Other), Foot (Left / Right / Both)
- `st.checkbox`: Champions League player?
- "Predict Value" button → big `st.metric` showing "Estimated Market Value: €XX.X Million"
- Show predictions from multiple models side by side

**"Find Undervalued Players" feature:**
- Table of players where predicted_value >> actual_value
- Columns: Player, Position, Club, Actual Value, Predicted Value, Difference (%)
- Sortable, with a narrative: "These players might be transfer bargains"

### Page 4: 🧠 Explainability (`pages/4_🧠_Explainability.py`)

Use SHAP library on the best model (likely XGBoost).

1. Global SHAP summary plot (beeswarm) — top 15 features
2. SHAP bar plot — mean absolute SHAP values
3. **Position-specific SHAP (the key insight):**
   - Filter data by position, run SHAP separately for Attackers, Midfielders, Defenders, GKs
   - Show side-by-side feature importance — "goals matter for strikers, league matters for defenders"
4. Individual player SHAP waterfall:
   - `st.selectbox` to pick any player → show waterfall explaining their valuation
5. SHAP dependence plots for top 3 features (age, goals_per_90, league_tier)

**Pre-compute SHAP values** and save as pickle. Load in Streamlit — never compute live.

### Page 5: ⚙️ Hyperparameter Tuning (`pages/5_⚙️_Hyperparameter_Tuning.py`)

Use Weights & Biases (`wandb`).

**Hyperparameters to tune:**
```
Ridge:          alpha ∈ [0.01, 0.1, 1, 10, 100]
Lasso:          alpha ∈ [0.001, 0.01, 0.1, 1, 10]
Decision Tree:  max_depth ∈ [3, 5, 10, 15, 20, None], min_samples_split ∈ [2, 5, 10, 20]
KNN:            n_neighbors ∈ [3, 5, 10, 20, 50], weights ∈ ['uniform', 'distance']
Random Forest:  n_estimators ∈ [50, 100, 200, 500], max_depth ∈ [5, 10, 15, 20, None]
XGBoost:        learning_rate ∈ [0.01, 0.05, 0.1, 0.2], max_depth ∈ [3, 5, 7, 10],
                n_estimators ∈ [100, 200, 300, 500], subsample ∈ [0.7, 0.8, 0.9, 1.0]
```

**W&B workflow:**
1. Create project "football-value-predictor"
2. Define sweep configs (Bayesian for XGBoost, grid for simpler models)
3. Log: hyperparameters, R², MAE, RMSE, training time per run
4. Select best run per model

**Streamlit page content:**
- Embedded W&B dashboard via `st.components.v1.iframe()` or screenshots
- Table: Model | Default R² | Tuned R² | Improvement (%) | Best Hyperparameters
- Before/after bar chart
- Learning curves (train vs validation vs model complexity)
- Narrative on which models improved most from tuning

---

## STEP 5: DEPLOYMENT

1. Push to GitHub
2. Deploy to Streamlit Cloud (share.streamlit.io)
3. Backup: HuggingFace Spaces with Streamlit SDK
4. Pre-train all models and save as .pkl — don't retrain on page load
5. Pre-compute SHAP values and save — don't recompute live

---

## REQUIREMENTS.TXT

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
wandb>=0.16.0
skrub>=0.3.0
joblib>=1.3.0
pyarrow>=14.0.0
```

Optional:
```
carte-ai>=0.1.0
torch>=2.0.0
```

---

## STYLE & UX

- Consistent football-themed color scheme (dark green + white + gold accents, or dark mode)
- Sidebar with navigation + project branding
- `st.metric` for key numbers, `st.columns` for layouts, `st.expander` for technical details
- `st.spinner` for any computation, `@st.cache_data` / `@st.cache_resource` for caching
- Keep everything fast — all heavy computation is pre-done and loaded from files

---

## ORDER OF IMPLEMENTATION

1. `src/data_loader.py` + `src/feature_engineering.py` → get the data pipeline working end-to-end
2. `src/models.py` → train all models, verify metrics, save .pkl files
3. Page 1 (Business Case) + Page 2 (Visualizations) → quickest to build
4. Page 3 (Predictions) → the main showcase
5. `src/shap_analysis.py` + Page 4 (Explainability) → the insight page
6. `src/wandb_tracking.py` + Page 5 (Hyperparameter Tuning) → the rigor page
7. Polish, deploy, test

---

## CRITICAL REMINDERS

- TARGET is `log_market_value = np.log1p(market_value_in_eur)`. Always train on log. Inverse-transform for display.
- Group Transfermarkt sub-positions (Centre-Forward, Left Winger, etc.) into 4 categories: Attack, Midfield, Defense, Goalkeeper.
- Filter to 2010+ data for cleaner valuations.
- Use most recent valuation per player per season if duplicates exist.
- Only compute per-90 stats for players with >= 450 minutes to avoid small-sample noise.
- Save ALL artifacts (models, SHAP values, processed data) as files. Never recompute in Streamlit.
- The app must have at least 2 models the user can toggle between on the prediction page (assignment requirement).
