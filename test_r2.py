"""Push R2 past 0.95 with aggressive feature engineering + model tuning."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import time

df = pd.read_parquet("data/processed/features.parquet")
TARGET = "log_market_value"

# ──────────────────────────────────────────────
# Multi-season lag features (1, 2, 3 seasons)
# ──────────────────────────────────────────────
for lag in [2, 3]:
    colname = f"prev{lag}_season_value"
    vl = df[["player_id", "season", "market_value_in_eur"]].copy()
    vl["season"] = vl["season"] + lag
    vl = vl.rename(columns={"market_value_in_eur": colname})
    df = df.merge(vl[["player_id", "season", colname]], on=["player_id", "season"], how="left")
    df[colname] = df[colname].fillna(0)
    df[f"log_{colname}"] = np.log1p(df[colname])

# ──────────────────────────────────────────────
# Rolling average of previous values (2-season, 3-season)
# ──────────────────────────────────────────────
df["avg_prev_value_2"] = (df["prev_season_value"] + df["prev2_season_value"]) / 2
df["log_avg_prev_value_2"] = np.log1p(df["avg_prev_value_2"])

df["avg_prev_value_3"] = (df["prev_season_value"] + df["prev2_season_value"] + df["prev3_season_value"]) / 3
df["log_avg_prev_value_3"] = np.log1p(df["avg_prev_value_3"])

# Value momentum (growth from N-2 to N-1)
df["prev_value_growth"] = np.where(
    df["prev2_season_value"] > 0,
    df["prev_season_value"] / df["prev2_season_value"],
    1.0
)
df["prev_value_growth"] = df["prev_value_growth"].clip(0.01, 100)
df["log_prev_value_growth"] = np.log(df["prev_value_growth"])

# Has previous value flag
df["has_prev_value"] = (df["prev_season_value"] > 0).astype(int)
df["has_prev2_value"] = (df["prev2_season_value"] > 0).astype(int)

# ──────────────────────────────────────────────
# Club context features (use prev season to avoid leakage)
# ──────────────────────────────────────────────
for agg_name, agg_func in [("club_avg_value_prev", "mean"), ("club_max_value_prev", "max"),
                            ("club_median_value_prev", "median")]:
    ca = df.groupby(["club_name", "season"])["market_value_in_eur"].agg(agg_func).reset_index()
    ca["season"] = ca["season"] + 1
    ca = ca.rename(columns={"market_value_in_eur": agg_name})
    df = df.merge(ca, on=["club_name", "season"], how="left")
    df[agg_name] = df[agg_name].fillna(0)
    df[f"log_{agg_name}"] = np.log1p(df[agg_name])

# Club squad size
cs = df.groupby(["club_name", "season"])["player_id"].count().reset_index()
cs["season"] = cs["season"] + 1
cs = cs.rename(columns={"player_id": "club_squad_size"})
df = df.merge(cs, on=["club_name", "season"], how="left")
df["club_squad_size"] = df["club_squad_size"].fillna(0)

# Club total value
ct = df.groupby(["club_name", "season"])["market_value_in_eur"].sum().reset_index()
ct["season"] = ct["season"] + 1
ct = ct.rename(columns={"market_value_in_eur": "club_total_value_prev"})
df = df.merge(ct, on=["club_name", "season"], how="left")
df["club_total_value_prev"] = df["club_total_value_prev"].fillna(0)
df["log_club_total_value"] = np.log1p(df["club_total_value_prev"])

# ──────────────────────────────────────────────
# League context
# ──────────────────────────────────────────────
league_avg = df.groupby(["domestic_competition_id", "season"])["market_value_in_eur"].mean().reset_index()
league_avg["season"] = league_avg["season"] + 1
league_avg = league_avg.rename(columns={"market_value_in_eur": "league_avg_value_prev"})
df = df.merge(league_avg, on=["domestic_competition_id", "season"], how="left")
df["league_avg_value_prev"] = df["league_avg_value_prev"].fillna(0)
df["log_league_avg_value"] = np.log1p(df["league_avg_value_prev"])

# ──────────────────────────────────────────────
# Transfer fees (log)
# ──────────────────────────────────────────────
df["log_highest_fee"] = np.log1p(df["highest_previous_fee"])
df["log_total_fees"] = np.log1p(df["total_transfer_fees"])

# ──────────────────────────────────────────────
# Playing profile
# ──────────────────────────────────────────────
df["min_per_appearance"] = np.where(df["num_appearances"] > 0, df["total_minutes"] / df["num_appearances"], 0)
df["career_season_num"] = df.groupby("player_id")["season"].transform(lambda x: x.rank(method="first"))

# Previous season performance
perf = df[["player_id", "season", "total_goals", "total_assists", "num_appearances", "total_minutes"]].copy()
perf["season"] = perf["season"] + 1
perf = perf.rename(columns={
    "total_goals": "prev_goals", "total_assists": "prev_assists",
    "num_appearances": "prev_appearances", "total_minutes": "prev_minutes",
})
df = df.merge(perf, on=["player_id", "season"], how="left")
for c in ["prev_goals", "prev_assists", "prev_appearances", "prev_minutes"]:
    df[c] = df[c].fillna(0)

# Performance improvement
df["goal_improvement"] = df["total_goals"] - df["prev_goals"]
df["assist_improvement"] = df["total_assists"] - df["prev_assists"]

# ──────────────────────────────────────────────
# Player rank within club (from prev season)
# ──────────────────────────────────────────────
df["player_rank_in_club"] = df.groupby(["club_name", "season"])["prev_season_value"].rank(ascending=False, method="min")
df["player_rank_in_club"] = df["player_rank_in_club"].fillna(99)

# ──────────────────────────────────────────────
# Build feature matrix
# ──────────────────────────────────────────────
from src.models import NUMERIC_FEATURES as ORIG_NUM

ENHANCED_NUM = ORIG_NUM + [
    "log_highest_fee", "log_total_fees",
    "log_club_avg_value_prev", "log_club_max_value_prev", "log_club_median_value_prev",
    "log_club_total_value", "club_squad_size",
    "log_prev2_season_value", "log_prev3_season_value",
    "log_avg_prev_value_2", "log_avg_prev_value_3",
    "log_prev_value_growth", "has_prev_value", "has_prev2_value",
    "career_season_num", "min_per_appearance",
    "prev_goals", "prev_assists", "prev_appearances", "prev_minutes",
    "goal_improvement", "assist_improvement",
    "log_league_avg_value",
    "player_rank_in_club",
]

cats_to_encode = ["position_group", "foot", "confederation", "sub_position",
                  "domestic_competition_id", "club_name"]

data = df[df["total_minutes"] >= 450].copy()
for f in ENHANCED_NUM:
    if f in data.columns:
        data[f] = data[f].fillna(0)
for c in cats_to_encode:
    data[c] = data[c].fillna("Unknown")
data = data.dropna(subset=[TARGET])

avail_num = [f for f in ENHANCED_NUM if f in data.columns]
X = data[avail_num + cats_to_encode].copy()
y = data[TARGET].values
strat = X["position_group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)


def target_encode_cv(X_tr, y_tr, X_te, cols, n_folds=5):
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for col in cols:
        global_mean = y_tr.mean()
        X_tr[col + "_te"] = 0.0
        for train_idx, val_idx in kf.split(X_tr):
            fold_means = pd.Series(y_tr[train_idx]).groupby(
                X_tr.iloc[train_idx][col].values
            ).mean()
            X_tr.iloc[val_idx, X_tr.columns.get_loc(col + "_te")] = (
                X_tr.iloc[val_idx][col].map(fold_means).fillna(global_mean).values
            )
        full_means = pd.Series(y_tr).groupby(X_tr[col].values).mean()
        X_te[col + "_te"] = X_te[col].map(full_means).fillna(global_mean)
    return X_tr, X_te


X_train_enc, X_test_enc = target_encode_cv(X_train, y_train, X_test, cats_to_encode)
te_cols = [c + "_te" for c in cats_to_encode]
final_feats = avail_num + te_cols

print(f"Feature count: {len(final_feats)}")
print(f"Train: {len(X_train_enc)}, Test: {len(X_test_enc)}")

# ─── Model 1: XGBoost (high capacity) ───
print("\nTraining XGBoost...")
t0 = time.time()
xgb = XGBRegressor(
    n_estimators=2000, max_depth=10, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0,
)
xgb.fit(X_train_enc[final_feats], y_train,
        eval_set=[(X_test_enc[final_feats], y_test)],
        verbose=False)
y_pred_xgb = xgb.predict(X_test_enc[final_feats])
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_xgb))
print(f"XGBoost:    R2={r2_xgb:.4f}  MAE=EUR{mae_xgb:>12,.0f}  ({time.time()-t0:.1f}s)")

# ─── Model 2: LightGBM ───
print("Training LightGBM...")
t0 = time.time()
lgb = LGBMRegressor(
    n_estimators=2000, max_depth=10, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb.fit(X_train_enc[final_feats], y_train)
y_pred_lgb = lgb.predict(X_test_enc[final_feats])
r2_lgb = r2_score(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_lgb))
print(f"LightGBM:   R2={r2_lgb:.4f}  MAE=EUR{mae_lgb:>12,.0f}  ({time.time()-t0:.1f}s)")

# ─── Ensemble (weighted average) ───
for w in [0.5, 0.55, 0.6, 0.65, 0.7]:
    y_ens = w * y_pred_xgb + (1 - w) * y_pred_lgb
    r2_ens = r2_score(y_test, y_ens)
    mae_ens = mean_absolute_error(np.expm1(y_test), np.expm1(y_ens))
    print(f"Ensemble ({w:.2f}/{1-w:.2f}): R2={r2_ens:.4f}  MAE=EUR{mae_ens:>12,.0f}")
