"""
Re-train MatchPredictor on fresh data, then run 10,000-game Monte Carlo.
Run from the wc2026_predictor directory:
    python3 retrain_and_simulate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config
from src.feature_engineering import build_training_matrix
from src.models import MatchPredictor, MonteCarloSimulator
from src.utils import get_logger

log = get_logger("retrain_and_simulate")


# ── Step 1: Rebuild training matrix ──────────────────────────────────────────
log.info("=== Step 1: Building training matrix ===")
df = build_training_matrix(min_year=1990, competitive_only=True)
df["date"] = pd.to_datetime(df["date"])

train_df = df[df["date"].dt.year <= 2014].copy()
val_df   = df[(df["date"].dt.year >= 2015) & (df["date"].dt.year <= 2018)].copy()
test_df  = df[df["date"].dt.year >= 2019].copy()

log.info("Train: %d  Val (2015-18): %d  Test (2019+): %d",
         len(train_df), len(val_df), len(test_df))

# ── Step 2: Retrain model ─────────────────────────────────────────────────────
log.info("=== Step 2: Training MatchPredictor ===")
predictor = MatchPredictor(random_state=config.RANDOM_STATE, n_optuna_trials=20)
predictor.fit(train_df, tune=True)

model_path = config.OUTPUTS_DIR / "match_predictor.pkl"
predictor.save(model_path)
log.info("Model saved → %s", model_path)

# Quick validation metrics
from sklearn.metrics import log_loss, accuracy_score
from src.models import _prepare_xy

for split_df, label in [(val_df, "Validation (2015-18)"), (test_df, "Test (2019+)")]:
    X, y_clf, _, _, _ = _prepare_xy(split_df)
    X_imp = predictor.imputer.transform(X)
    X_sc  = predictor.scaler.transform(X_imp)
    proba = predictor.xgb_model.predict_proba(X_sc)
    ll  = log_loss(y_clf, proba)
    acc = accuracy_score(y_clf, proba.argmax(axis=1))
    log.info("%s  log_loss=%.4f  accuracy=%.4f", label, ll, acc)

# ── Step 3: Monte Carlo (10,000 simulations) ──────────────────────────────────
log.info("=== Step 3: Monte Carlo (%d simulations) ===", config.N_SIMULATIONS)
sim = MonteCarloSimulator(
    predictor=predictor,
    n_simulations=config.N_SIMULATIONS,
    random_state=config.RANDOM_STATE,
)
sim.precompute_lambda_cache()

mc_results = sim.run_full_tournament(verbose=True)

results_df = (
    pd.DataFrame(mc_results).T
    .reset_index()
    .rename(columns={"index": "team"})
    .sort_values("win_pct", ascending=False)
    .reset_index(drop=True)
)
for col in ["win_pct", "final_pct", "sf_pct", "qf_pct", "r16_pct", "group_exit_pct"]:
    results_df[col] = (results_df[col] * 100).round(2)

out_path = config.OUTPUTS_DIR / "mc_tournament_results.csv"
results_df.to_csv(out_path, index=False)
log.info("MC results saved → %s", out_path)

print("\n=== Top 16 Teams by Championship Probability ===")
print(
    results_df[["team", "win_pct", "final_pct", "sf_pct", "qf_pct", "exit_stage"]]
    .head(16)
    .to_string(index=False)
)
