# ⚽ Football Player Market Value Predictor
## Full Project Plan — Streamlit App

---

## 1. THE PITCH (Business Case)

**Problem:** Football clubs spend billions on transfers every year, often overpaying or missing undervalued talent. Can we build a data-driven tool that estimates a player's market value based on their performance stats, profile, and context — essentially a "Moneyball for football"?

**Target Variable:** `market_value_in_eur` (continuous, from `player_valuations` table)

**Who would use this?** Scouts, sporting directors, agents, fantasy football analysts, fans.

---

## 2. DATASET ARCHITECTURE

Source: [Transfermarkt Kaggle Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores) — 10 relational tables.

### Tables You'll Use

| Table | What You Extract |
|---|---|
| `players` | age, position, nationality, height, foot, current_club_id |
| `appearances` | goals, assists, minutes_played, yellow_cards, red_cards per season |
| `player_valuations` | market_value_in_eur over time (YOUR TARGET) |
| `games` | competition_id, home/away context |
| `competitions` | league tier / prestige (Premier League vs. 2. Bundesliga) |
| `clubs` | club size (stadium_seats as proxy for club prestige) |
| `transfers` | transfer_fee history, number of moves |
| `game_events` | goal types, assists breakdown |

### Feature Engineering Pipeline

This is where you show real pandas skill. You'll join multiple tables and engineer features like:

```
Per-Season Aggregated Stats (from appearances):
  - total_goals, total_assists, total_minutes
  - goals_per_90, assists_per_90 (normalized)
  - yellow_cards_per_90, red_cards_per_90
  - games_started vs games_as_sub

Player Profile (from players):
  - age (at time of valuation)
  - position (one-hot or grouped: Attack/Midfield/Defense/Goalkeeper)
  - foot (left/right/both)
  - nationality_confederation (UEFA, CONMEBOL, etc.)

Club & League Context:
  - league_tier (top 5 leagues vs. others)
  - club_prestige (stadium_seats as proxy, or avg squad value)
  - champions_league_appearances (binary or count)

Transfer History (from transfers):
  - num_previous_transfers
  - highest_previous_fee
  - years_at_current_club

Derived / Interaction Features:
  - goal_involvement = goals + assists
  - minutes_per_goal_involvement
  - age_squared (captures non-linear aging effect)
  - is_peak_age (25-29 flag)
```

### Target Variable Transformation

Market values are heavily right-skewed (most players worth <5M, a few worth 100M+).
**Use log transformation:** `y = np.log1p(market_value_in_eur)` → makes regression much more effective.
Inverse-transform predictions for display: `np.expm1(y_pred)`.

---

## 3. MODELS (The 5+ Required + Extras)

Your assignment says "at least 5 models." Here's how to structure them to show range AND tell a coherent story:

### Model Lineup

| # | Model | Library | Why It's Here | Role in Story |
|---|---|---|---|---|
| 1 | **Linear Regression (OLS)** | sklearn | Baseline — required by assignment | "How far can simplicity take us?" |
| 2 | **Ridge Regression** | sklearn | L2 regularization handles multicollinearity | "What if we regularize?" |
| 3 | **Lasso Regression** | sklearn | L1 regularization → automatic feature selection | "Which features actually matter?" |
| 4 | **Decision Tree Regressor** | sklearn | Non-linear, interpretable | "What if the relationships aren't linear?" |
| 5 | **KNN Regressor** | sklearn | Instance-based, captures local patterns | "Similar players should have similar values" |
| 6 | **XGBoost Regressor** | xgboost | State-of-the-art gradient boosting | "The industry standard for tabular data" |
| 7 | **Random Forest** | sklearn | Ensemble of trees, robust | Comparison point for XGBoost |

### Bonus / Extra Mile Models

| # | Model | Library | Why It's Impressive |
|---|---|---|---|
| 8 | **skrub + tabular_learner** | skrub | Shows you know cutting-edge preprocessing tools. Uses `TableVectorizer` to auto-handle dirty categorical data (player names, club names) and `tabular_learner('regressor')` for instant strong baselines. |
| 9 | **CARTE Regressor** | carte-ai | Pretrained tabular model — this is genuinely cutting-edge (2024 paper from INRIA). Treats each row as a graph. If your professor or an interviewer sees this, it shows you're reading research. |

### Important Note on Logistic Regression
Your assignment says **linear regression**. Logistic regression is for classification, not regression. You could include it IF you reframe a sub-problem as classification (e.g., "Is this player worth >50M? Yes/No") — but don't make it a main model. Instead, mention it on the explainability page as an alternative framing.

### How Models Connect on the Streamlit Page

The **Prediction Page** should let users toggle between at least 2 models (as required). Best approach:
- Dropdown: "Choose model" → shows prediction + confidence
- Side-by-side comparison table of all models' metrics (R², MAE, RMSE)
- A chart showing predicted vs actual scatter plot per model

---

## 4. STREAMLIT APP — PAGE BY PAGE

### Page 1: 🏠 Business Case & Data Presentation

**Content:**
- The problem statement ("How much is a player worth?")
- Why this matters (transfer market size, overpaying examples)
- Dataset overview: source, size, table structure (show the ER diagram)
- Sample data preview (interactive `st.dataframe`)
- Key stats: number of players, leagues covered, date range
- Data quality: missing values heatmap, distributions of key columns

**Extra mile:**
- Interactive entity-relationship diagram showing how the 10 tables connect
- A "fun fact" section: most expensive player, biggest overpay, etc.

---

### Page 2: 📊 Data Visualization & Insights

**Content (aim for 6-8 compelling charts):**

1. **Market value distribution** — histogram + log-transformed version (show why you transform)
2. **Value by position** — box plot showing Attackers > Midfielders > Defenders > GKs
3. **Age curve** — average market value by age (the famous "peak at 27" curve)
4. **League premium** — bar chart of average player value by league (PL >> others)
5. **Goals/Assists vs Value** — scatter plot with regression line (your core relationship)
6. **Correlation heatmap** — all numeric features vs market value
7. **Top clubs by total squad value** — horizontal bar chart
8. **Value inflation over time** — line chart showing market values increasing year over year

**Libraries:** Seaborn + Matplotlib (as required). Optionally Plotly for interactivity.

**Extra mile:**
- Let users filter by position/league and watch charts update dynamically
- Add a "Player Lookup" widget: type a name, see their value trajectory over time

---

### Page 3: 🔮 Prediction Page

**Content:**
- **Model selector:** Dropdown to switch between models (minimum 2 required, show all 7+)
- **Interactive prediction form:**
  - Sliders: age, goals, assists, minutes_played, games
  - Dropdowns: position, league, foot
  - Output: "Estimated market value: €XX million"
- **Model comparison dashboard:**
  - Table: Model | R² | MAE | RMSE | Training Time
  - Bar chart of R² scores across all models
  - Predicted vs Actual scatter plot (toggle per model)
  - Residual distribution plot

**The "wow" feature:**
- "Find Undervalued Players" button — shows players where predicted value >> actual value
- Position-specific predictions: "As a striker with these stats, you're worth X. As a midfielder, you'd be worth Y."

---

### Page 4: 🧠 Explainable AI (SHAP)

**Library:** SHAP (recommended over Shapash for this — more flexible, better plots)

**Content:**
- **Global feature importance:** SHAP summary plot showing which features drive value most
- **SHAP beeswarm plot:** Shows direction + magnitude of each feature's impact
- **Position-specific SHAP:** Run separate explanations for Attackers vs Defenders vs Goalkeepers — the driving features will be different, and that's the insight
- **Individual player breakdown:** SHAP waterfall plot for a specific player (e.g., "Why is Haaland valued at €180M?")
- **Feature dependence plots:** SHAP dependence for top 3 features (age, goals, league)

**Narrative to tell:**
- "For attackers, goals_per_90 is the #1 driver. For defenders, it's league prestige."
- "Age has a non-linear effect — value peaks at 27 and drops sharply after 30."
- "Playing in the Premier League adds ~€X million to a player's estimated value."

**Extra mile:**
- Interactive: let user select any player from a dropdown and see their SHAP waterfall
- Compare SHAP between two models (e.g., XGBoost vs Linear) — shows how different models "think"

---

### Page 5: ⚙️ Hyperparameter Tuning (W&B)

**Library:** Weights & Biases (`wandb`)

**How it works:**
1. Create a W&B project (free tier is fine)
2. For each model, run hyperparameter sweeps using `wandb.sweep()`
3. Log metrics (R², MAE, RMSE) + hyperparameters to W&B
4. Embed the W&B dashboard in Streamlit using `st.components.v1.iframe()`

**What to tune:**

```
Ridge/Lasso:     alpha (regularization strength)
Decision Tree:   max_depth, min_samples_split, min_samples_leaf
KNN:             n_neighbors, weights, metric
Random Forest:   n_estimators, max_depth, min_samples_split, max_features
XGBoost:         learning_rate, max_depth, n_estimators, subsample, colsample_bytree
```

**Page content:**
- Embedded W&B dashboard (or screenshots if embedding is tricky)
- Table of best hyperparameters per model
- Before/after comparison: default params vs tuned params
- Learning curves showing overfitting/underfitting
- A narrative: "XGBoost with tuned parameters improved R² from 0.82 to 0.89"

**Extra mile:**
- Show a parallel coordinates plot of hyperparameter combinations
- Run a `wandb.sweep` with Bayesian optimization (not just grid/random search)
- Log training curves, not just final metrics

---

## 5. TECH STACK SUMMARY

| Component | Tool |
|---|---|
| App framework | Streamlit |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn (required), + Plotly (extra) |
| ML Models | scikit-learn, XGBoost |
| Preprocessing | skrub (TableVectorizer, tabular_learner) |
| Advanced model | carte-ai (CARTERegressor) |
| Explainability | SHAP |
| Experiment tracking | Weights & Biases |
| Deployment | Streamlit Cloud + HuggingFace Spaces |

---

## 6. PROJECT STRUCTURE

```
football-value-predictor/
├── app.py                          # Main Streamlit entry point (multipage)
├── pages/
│   ├── 1_🏠_Business_Case.py
│   ├── 2_📊_Visualizations.py
│   ├── 3_🔮_Predictions.py
│   ├── 4_🧠_Explainability.py
│   └── 5_⚙️_Hyperparameter_Tuning.py
├── src/
│   ├── data_loader.py              # Load + join the 10 tables
│   ├── feature_engineering.py      # All feature creation logic
│   ├── models.py                   # Train/evaluate all models
│   ├── shap_analysis.py            # SHAP computation helpers
│   └── wandb_tracking.py           # W&B logging utilities
├── data/
│   ├── raw/                        # Original CSVs from Kaggle
│   └── processed/                  # Engineered feature matrix
├── models/                         # Saved trained models (.pkl)
├── notebooks/
│   └── exploration.ipynb           # Initial EDA (not shown in app)
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml                 # Theme configuration
```

---

## 7. TIMELINE MAPPING (to your assignment deadlines)

| Deadline | Deliverable | What to Do |
|---|---|---|
| **Wednesday (Week 1)** | Team + Dataset + Problem | Download dataset, validate it works, define problem statement |
| **Monday (Week 2)** | Landing page + Data description | Build Page 1, write data_loader.py, feature_engineering.py |
| **Wednesday (Week 2)** | Data Visualization page | Build Page 2 — all 6-8 charts |
| **Monday (Week 3)** | Prediction / Forecast page | Train all models, build Page 3 with model comparison |
| **Monday (Week 4)** | Explainability + Hyperparameter tuning | Build Pages 4 & 5, run W&B sweeps, SHAP analysis |
| **Wednesday (Week 4)** | Speech Day | Deploy to Streamlit Cloud, rehearse 8-min presentation |

**Tip:** Deploy early (even a skeleton) to Streamlit Cloud and HuggingFace. Don't leave deployment to the last day.

---

## 8. PRESENTATION STRATEGY (8 minutes)

Suggested split (adjust for team size):

| Time | Speaker | Content |
|---|---|---|
| 0:00-1:30 | Person 1 | Problem intro, why it matters, dataset overview (Page 1) |
| 1:30-3:00 | Person 2 | Key insights from visualizations — the "age curve," league premium (Page 2) |
| 3:00-5:00 | Person 3 | Model comparison, live demo of prediction tool, best model results (Page 3) |
| 5:00-6:30 | Person 4 | SHAP analysis — "what drives value for attackers vs defenders" (Page 4) |
| 6:30-7:30 | Person 5 | Hyperparameter tuning journey, how we improved the model (Page 5) |
| 7:30-8:00 | Anyone | Conclusion, limitations, what we'd do next |

**Key presentation tips:**
- LIVE DEMO the prediction page — have an interviewer/professor type in stats and see a prediction
- Have a "surprising finding" ready: e.g., "We found that playing for a Premier League club adds €12M to a player's estimated value regardless of performance"
- End with limitations honestly: "Market value is also driven by marketability, social media following, and agent negotiations — factors not in our data"

---

## 9. WHAT MAKES THIS "EXTRA MILE" 🏃

1. **skrub** — Almost no student projects use this. Shows you know the cutting-edge sklearn ecosystem.
2. **CARTE** — A 2024 research paper's model. If you can get it running, it's genuinely impressive.
3. **Multi-table joins** — You're not just loading one CSV. You're doing real data engineering across 10 related tables.
4. **Log-transformed target** — Shows you understand regression assumptions.
5. **Position-specific SHAP** — Not just "feature importance" but "feature importance *changes by context*."
6. **Undervalued player finder** — Turns the model into a product, not just an exercise.
7. **W&B with Bayesian sweeps** — Goes beyond basic grid search.

---

## 10. RISKS & MITIGATIONS

| Risk | Mitigation |
|---|---|
| CARTE is hard to install (PyTorch + graph libs) | Have it as a bonus. If it fails, you still have 7 models. |
| Dataset is large, Streamlit is slow | Pre-compute and cache everything with `@st.cache_data`. Save trained models as .pkl files. |
| Market value is hard to predict (R² might be modest) | Frame it: "Even 70% R² is useful for initial screening." Log-transform helps a lot. |
| W&B embedding in Streamlit is finicky | Fallback: take screenshots of W&B dashboard and display them. |
| Team coordination on code | Use Git branches. One person owns data pipeline, one owns models, one owns Streamlit pages. |
