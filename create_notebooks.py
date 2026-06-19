"""Helper script to create all 7 Jupyter notebooks programmatically."""
import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


# ─────────────────────────────────────────────────────────────────────────────
# 01_data_collection.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb01 = nb([
    md("# Notebook 01 — Data Collection\nDownload and cache all 10 data sources."),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from src.data_collection import (
    fetch_international_results,
    fetch_goalscorers,
    fetch_shootouts,
    fetch_elo_ratings,
    fetch_elo_history,
    fetch_fifa_rankings,
    fetch_transfermarkt_squads,
    fetch_fifa_player_ratings,
    build_squad_features,
    fetch_statsbomb_aggregates,
    fetch_manager_data,
    fetch_socioeconomic,
    fetch_venue_data,
    compute_travel_distances,
    compute_h2h_stats,
    fetch_wc2026_fixtures,
)
from src.utils import get_logger
log = get_logger("01_data_collection")
"""),
    md("## 1. International Match Results"),
    code("""
results = fetch_international_results(force_refresh=False)
print(f"Results shape: {results.shape}")
print(f"Date range: {results['date'].min()} → {results['date'].max()}")
print(f"Competitive: {results['competitive'].sum()} / {len(results)}")
results.head()
"""),
    md("## 2. Elo Ratings"),
    code("""
elo = fetch_elo_ratings()
print(f"Elo rows: {len(elo)}")
elo.sort_values("elo_rating", ascending=False).head(10)
"""),
    code("""
elo_hist = fetch_elo_history()
print(f"Elo history rows: {len(elo_hist)}")
elo_hist.head()
"""),
    md("## 3. FIFA Rankings"),
    code("""
rankings = fetch_fifa_rankings()
print(f"Rankings rows: {len(rankings)}")
rankings.head(10)
"""),
    md("## 4. Squad & Player Quality"),
    code("""
squads = fetch_transfermarkt_squads()
ratings = fetch_fifa_player_ratings()
squad_features = build_squad_features()
print(f"Squad features shape: {squad_features.shape}")
squad_features.sort_values("top11_fifa_overall_avg", ascending=False).head(10)
"""),
    md("## 5. StatsBomb Aggregates"),
    code("""
sb = fetch_statsbomb_aggregates()
print(f"StatsBomb rows: {len(sb)}")
sb.sort_values("avg_xg_for", ascending=False).head(10)
"""),
    md("## 6. Manager Data"),
    code("""
managers = fetch_manager_data()
print(f"Managers: {len(managers)}")
managers[["team", "manager_name", "win_rate", "tournament_matches"]].head(10)
"""),
    md("## 7. Socioeconomic Data"),
    code("""
soc = fetch_socioeconomic()
print(f"Socioeconomic rows: {len(soc)}")
soc.sort_values("gdp_per_capita_usd", ascending=False).head(10)
"""),
    md("## 8. Venue & Geography"),
    code("""
venues = fetch_venue_data()
distances = compute_travel_distances()
print(f"Venues: {len(venues)}")
print(f"Distance rows: {len(distances)}")
venues[["city", "country", "stadium", "altitude_m", "capacity"]]
"""),
    md("## 9. Head-to-Head History"),
    code("""
h2h = compute_h2h_stats(results=results, force_refresh=False)
print(f"H2H pairs: {len(h2h)}")
h2h[h2h["h2h_matches"] > 0].sort_values("h2h_matches", ascending=False).head(10)
"""),
    md("## 10. 2026 WC Fixtures"),
    code("""
fixtures = fetch_wc2026_fixtures()
print(f"Group-stage fixtures: {len(fixtures)}")
fixtures.head(20)
"""),
    md("## Summary"),
    code("""
import pandas as pd
summary = {
    "results": len(results),
    "elo_current": len(elo),
    "elo_history": len(elo_hist),
    "rankings": len(rankings),
    "squad_features": len(squad_features),
    "statsbomb": len(sb),
    "managers": len(managers),
    "socioeconomic": len(soc),
    "venues": len(venues),
    "h2h_pairs": len(h2h),
    "fixtures": len(fixtures),
}
pd.Series(summary, name="rows").to_frame()
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# 02_eda.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb02 = nb([
    md("# Notebook 02 — Exploratory Data Analysis"),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.data_collection import (
    fetch_international_results, fetch_elo_ratings,
    fetch_fifa_rankings, build_squad_features, fetch_wc2026_fixtures,
)
import config

plt.rcParams["figure.figsize"] = (12, 5)
sns.set_theme(style="whitegrid")

results = fetch_international_results()
elo = fetch_elo_ratings()
squad = build_squad_features()
fixtures = fetch_wc2026_fixtures()
"""),
    md("## Results Distribution"),
    code("""
results["date"] = pd.to_datetime(results["date"])
results["year"] = results["date"].dt.year
results["home_goals"] = pd.to_numeric(results["home_score"], errors="coerce")
results["away_goals"] = pd.to_numeric(results["away_score"], errors="coerce")
results["total_goals"] = results["home_goals"] + results["away_goals"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
results["home_goals"].plot.hist(bins=10, ax=axes[0], title="Home Goals Distribution", edgecolor="black")
results["away_goals"].plot.hist(bins=10, ax=axes[1], title="Away Goals Distribution", edgecolor="black", color="orange")
results["total_goals"].plot.hist(bins=15, ax=axes[2], title="Total Goals Distribution", edgecolor="black", color="green")
plt.tight_layout(); plt.show()
"""),
    code("""
# Win/Draw/Loss rates
competitive = results[results["competitive"] == True].copy()
competitive["result"] = np.where(
    competitive["home_goals"] > competitive["away_goals"], "Home Win",
    np.where(competitive["home_goals"] == competitive["away_goals"], "Draw", "Away Win")
)
result_counts = competitive["result"].value_counts()
px.pie(values=result_counts.values, names=result_counts.index,
       title="Competitive International Results (Home/Draw/Away)").show()
"""),
    md("## Goals Per Year Trend"),
    code("""
yearly = competitive.groupby("year")["total_goals"].mean().reset_index()
yearly.columns = ["year", "avg_goals"]
fig = px.line(yearly, x="year", y="avg_goals",
              title="Average Goals per Competitive Match by Year")
fig.show()
"""),
    md("## Elo Rating Distribution (WC 2026 Teams)"),
    code("""
elo_wc = pd.DataFrame([
    {"team": t, "elo": e, "confederation": config.CONFEDERATION_MAP.get(t, "Other")}
    for t, e in config.INITIAL_ELO.items()
]).sort_values("elo", ascending=False)

fig = px.bar(elo_wc, x="team", y="elo", color="confederation",
             title="Elo Ratings — 2026 WC Teams",
             labels={"elo": "Elo Rating", "team": "Team"})
fig.update_xaxes(tickangle=45)
fig.show()
"""),
    md("## Squad Market Value vs FIFA Ranking"),
    code("""
rankings_dict = config.FIFA_RANKINGS
squad_plot = squad.copy()
squad_plot["fifa_rank"] = squad_plot["team"].map(rankings_dict)
squad_plot["confederation"] = squad_plot["team"].map(config.CONFEDERATION_MAP)

fig = px.scatter(
    squad_plot, x="fifa_rank", y="avg_squad_market_value_m",
    hover_name="team", color="confederation",
    size="top11_fifa_overall_avg",
    title="Squad Market Value vs FIFA Ranking",
    labels={"fifa_rank": "FIFA Ranking", "avg_squad_market_value_m": "Avg Market Value (€M)"},
)
fig.show()
"""),
    md("## Confederation Goal Scoring Rates"),
    code("""
results["confederation_home"] = results["home_team"].map(config.CONFEDERATION_MAP)
conf_goals = results.groupby("confederation_home")[["home_goals", "away_goals"]].mean()
conf_goals.columns = ["avg_goals_scored", "avg_goals_conceded"]
conf_goals = conf_goals.dropna()

conf_goals.plot.bar(figsize=(10, 5), title="Average Goals by Confederation (home perspective)")
plt.ylabel("Average Goals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""),
    md("## Elo Correlation with Match Results"),
    code("""
# Merge Elo into results
elo_dict = dict(zip(elo["team"], elo["elo_rating"]))
sample = results.sample(min(2000, len(results)), random_state=42).copy()
sample["elo_home"] = sample["home_team"].map(elo_dict)
sample["elo_away"] = sample["away_team"].map(elo_dict)
sample["elo_diff"] = sample["elo_home"] - sample["elo_away"]
sample["home_win"] = (sample["home_goals"] > sample["away_goals"]).astype(int)
sample = sample.dropna(subset=["elo_diff"])

fig, ax = plt.subplots(figsize=(10, 5))
sample["elo_bin"] = pd.cut(sample["elo_diff"], bins=10)
win_rate = sample.groupby("elo_bin")["home_win"].mean()
win_rate.plot.bar(ax=ax, title="Home Win Rate by Elo Difference", edgecolor="black")
ax.set_ylabel("Home Win Rate")
ax.axhline(0.5, color="red", linestyle="--", label="50% baseline")
ax.legend()
plt.tight_layout()
plt.show()
"""),
    md("## 2026 WC Groups Overview"),
    code("""
groups_data = []
for group, teams in config.GROUPS.items():
    for team in teams:
        groups_data.append({
            "group": group, "team": team,
            "elo": config.INITIAL_ELO.get(team, 1500),
            "fifa_rank": config.FIFA_RANKINGS.get(team, 48),
            "confederation": config.CONFEDERATION_MAP.get(team, "Other"),
        })
groups_df = pd.DataFrame(groups_data)

fig = px.scatter(
    groups_df, x="group", y="elo", color="confederation",
    hover_name="team", size="elo",
    title="2026 WC Group Compositions by Elo Rating",
    labels={"group": "Group", "elo": "Elo Rating"},
)
fig.show()
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# 03_feature_engineering.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb03 = nb([
    md("# Notebook 03 — Feature Engineering"),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.feature_engineering import build_match_features, build_training_matrix
import config
"""),
    md("## Single match feature vector"),
    code("""
fv = build_match_features(
    team1="Spain",
    team2="Morocco",
    date="2026-06-15",
    venue="Estadio Azteca",
    tournament_stage="group",
)
print(f"Feature vector size: {len(fv)}")
pd.Series(fv).to_frame("value").head(40)
"""),
    md("## Key feature differences: strong vs weak team"),
    code("""
fv_strong = build_match_features("Spain", "New Caledonia", "2026-06-15", "MetLife Stadium", "group")
fv_weak   = build_match_features("New Caledonia", "Spain", "2026-06-15", "MetLife Stadium", "group")

comparison = pd.DataFrame({
    "Spain vs NC": pd.Series(fv_strong),
    "NC vs Spain": pd.Series(fv_weak),
})
key_features = ["elo_rating_t1", "elo_rating_t2", "elo_diff",
                "top11_fifa_overall_avg_t1", "top11_fifa_overall_avg_t2",
                "avg_squad_market_value_m_t1", "avg_squad_market_value_m_t2",
                "h2h_win_rate_t1", "form_last10_points_t1", "form_last10_points_t2"]
comparison.loc[key_features]
"""),
    md("## Build full training matrix"),
    code("""
train_df = build_training_matrix(min_year=2000, competitive_only=True, force_refresh=False)
print(f"Training matrix: {train_df.shape}")
print(f"Result distribution:\\n{train_df['result'].value_counts()}")
train_df.head()
"""),
    md("## Feature correlation heatmap"),
    code("""
feat_cols = [c for c in train_df.columns
             if c not in {"result","goals_t1","goals_t2","date","team1","team2"}]
corr_target = train_df[feat_cols + ["result"]].corr()["result"].drop("result")
top_corr = corr_target.abs().sort_values(ascending=False).head(25)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_corr.values, y=top_corr.index, palette="RdYlGn")
plt.title("Top 25 Features by Absolute Correlation with Match Result")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.show()
"""),
    md("## Missing value analysis"),
    code("""
missing = train_df[feat_cols].isnull().sum()
missing_pct = (missing / len(train_df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
if not missing_pct.empty:
    missing_pct.plot.barh(title="Missing Value % per Feature")
    plt.tight_layout()
    plt.show()
else:
    print("No missing values in feature matrix!")
"""),
    md("## Feature distributions"),
    code("""
key_plot_features = [
    "elo_diff", "market_value_ratio", "form_diff",
    "h2h_win_rate_t1", "gdp_per_capita_usd_t1",
]
fig, axes = plt.subplots(1, len(key_plot_features), figsize=(18, 4))
for ax, feat in zip(axes, key_plot_features):
    if feat in train_df.columns:
        train_df[feat].dropna().hist(bins=30, ax=ax, edgecolor="black")
        ax.set_title(feat, fontsize=9)
plt.tight_layout()
plt.show()
"""),
    md("## Save processed training matrix"),
    code("""
import config
out_path = config.PROCESSED_DIR / "training_matrix.csv"
train_df.to_csv(out_path, index=False)
print(f"Saved → {out_path}  ({train_df.shape})")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# 04_ml_model.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb04 = nb([
    md("# Notebook 04 — ML Model Training & Evaluation"),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

import config
from src.feature_engineering import build_training_matrix
from src.models import MatchPredictor, _get_feature_cols, _prepare_xy
from src.utils import get_logger

log = get_logger("04_ml_model")
"""),
    md("## Load training data & temporal splits"),
    code("""
df = build_training_matrix(min_year=1990, competitive_only=True)
df["date"] = pd.to_datetime(df["date"])

train_df = df[df["date"].dt.year <= 2014].copy()
val_df   = df[(df["date"].dt.year >= 2015) & (df["date"].dt.year <= 2018)].copy()
test_df  = df[df["date"].dt.year >= 2019].copy()

print(f"Train: {len(train_df)}, Val (2015-18): {len(val_df)}, Test (2019+): {len(test_df)}")
"""),
    md("## Train MatchPredictor (XGBoost + LightGBM + CatBoost ensemble)"),
    code("""
predictor = MatchPredictor(random_state=config.RANDOM_STATE, n_optuna_trials=20)
predictor.fit(train_df, tune=True)
print("Model fitted successfully.")
"""),
    md("## Validation metrics (WC 2018 equivalent)"),
    code("""
def evaluate(predictor, df, label):
    X, y_clf, y_g1, y_g2, fcols = _prepare_xy(df)
    X_imp = predictor.imputer.transform(X)
    X_sc  = predictor.scaler.transform(X_imp)
    proba_xgb = predictor.xgb_model.predict_proba(X_sc)
    y_true_bin = (y_clf == 2).astype(int)
    proba_win = proba_xgb[:, 2]
    ll = log_loss(y_clf, proba_xgb)
    acc = accuracy_score(y_clf, proba_xgb.argmax(axis=1))
    bs = brier_score_loss(y_true_bin, proba_win)
    print(f"\\n{label}")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Brier Score: {bs:.4f}")
    return proba_xgb, y_clf

val_proba, val_y = evaluate(predictor, val_df, "Validation (2015-2018)")
test_proba, test_y = evaluate(predictor, test_df, "Test (2019+)")
"""),
    md("## Calibration curves"),
    code("""
fig, ax = plt.subplots(figsize=(8, 6))
for proba, y, label in [(val_proba, val_y, "Validation"), (test_proba, test_y, "Test")]:
    y_bin = (y == 2).astype(int)
    fraction, mean_pred = calibration_curve(y_bin, proba[:, 2], n_bins=10)
    ax.plot(mean_pred, fraction, marker="o", label=label)
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability (win)")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve — Win Probability")
ax.legend()
plt.tight_layout()
plt.show()
"""),
    md("## Feature Importance (XGBoost)"),
    code("""
feat_importance = pd.Series(
    predictor.xgb_model.feature_importances_,
    index=predictor.feature_cols,
).sort_values(ascending=False)

top20 = feat_importance.head(20)
plt.figure(figsize=(10, 7))
sns.barplot(x=top20.values, y=top20.index, palette="Blues_r")
plt.title("XGBoost Feature Importance (Top 20)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
print("Top 20 features:\\n", top20.to_string())
"""),
    md("## SHAP values"),
    code("""
try:
    import shap
    X_test, _, _, _, _ = _prepare_xy(test_df.sample(min(500, len(test_df)), random_state=42))
    shap_vals = predictor.shap_values(X_test)
    # For multiclass, shap_vals is a list; plot class 2 (win_t1)
    sv = shap_vals[2] if isinstance(shap_vals, list) else shap_vals
    shap.summary_plot(sv, X_test, feature_names=predictor.feature_cols, max_display=20, show=True)
except Exception as e:
    print(f"SHAP plot skipped: {e}")
"""),
    md("## Save model"),
    code("""
model_path = config.OUTPUTS_DIR / "match_predictor.pkl"
predictor.save(model_path)
print(f"Model saved → {model_path}")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# 05_monte_carlo.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb05 = nb([
    md("# Notebook 05 — Monte Carlo Tournament Simulation"),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm

import config
from src.models import MatchPredictor, MonteCarloSimulator
from src.utils import get_logger

log = get_logger("05_monte_carlo")
"""),
    md("## Load trained predictor"),
    code("""
model_path = config.OUTPUTS_DIR / "match_predictor.pkl"
try:
    predictor = MatchPredictor.load(model_path)
    print(f"Loaded model from {model_path}")
except Exception as e:
    print(f"Could not load model ({e}), using Elo-based fallback")
    predictor = None
"""),
    md("## Single match simulation"),
    code("""
sim = MonteCarloSimulator(predictor=predictor, n_simulations=1000, random_state=42)

match_result = sim.simulate_match("Spain", "France", venue="MetLife Stadium", stage="sf", allow_draw=False)

print(f"\\nSpain vs France (SF)")
print(f"Score: {match_result['score']['team1']} - {match_result['score']['team2']}")
print(f"Winner: {match_result['winner']}")
print(f"Extra time: {match_result['extra_time']}, Penalties: {match_result['penalties']}")
print(f"xG: {match_result['xg']}")
print(f"Possession: {match_result['possession']}")
print(f"Strategy: {match_result['strategy']}")
print(f"\\nGoal scorers:")
for g in match_result["goal_scorers"]:
    print(f"  {g['minute']}' {g['player']} ({g['team']}) [{g['type']}]")
print(f"\\nNarrative:\\n{match_result['narrative']}")
"""),
    md("## Group stage simulation"),
    code("""
print("Simulating group stage...")
standings = sim.simulate_group_stage()

for group in sorted(standings.keys()):
    df = standings[group]
    print(f"\\nGroup {group}:")
    print(df[["team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]].to_string(index=False))
"""),
    md("## Penalty shootout simulation"),
    code("""
print("Penalty shootout: Brazil vs France")
s1, s2, winner = sim.simulate_penalty_shootout("Brazil", "France")
print(f"Score: Brazil {s1} - {s2} France | Winner: {winner}")

# Run 1000 shootouts to get probabilities
brazil_wins = 0
for _ in range(1000):
    _, _, w = sim.simulate_penalty_shootout("Brazil", "France")
    if w == "Brazil":
        brazil_wins += 1
print(f"\\nBrazil wins shootout in {brazil_wins/10:.1f}% of simulations")
"""),
    md("## Full tournament simulation (10,000 runs)"),
    code("""
sim_full = MonteCarloSimulator(predictor=predictor, n_simulations=config.N_SIMULATIONS, random_state=42)
print(f"Running {config.N_SIMULATIONS:,} tournament simulations...")
mc_results = sim_full.run_full_tournament(verbose=True)

results_df = pd.DataFrame(mc_results).T.reset_index()
results_df.columns = ["team"] + list(results_df.columns[1:])
results_df = results_df.sort_values("win_pct", ascending=False).reset_index(drop=True)
results_df["win_pct"] = (results_df["win_pct"] * 100).round(1)
results_df["final_pct"] = (results_df["final_pct"] * 100).round(1)
results_df["sf_pct"] = (results_df["sf_pct"] * 100).round(1)
results_df["qf_pct"] = (results_df["qf_pct"] * 100).round(1)

print("\\nTop 16 teams by tournament win probability:")
print(results_df[["team", "win_pct", "final_pct", "sf_pct", "qf_pct", "exit_stage"]].head(16).to_string(index=False))

# Save
results_df.to_csv(config.OUTPUTS_DIR / "mc_tournament_results.csv", index=False)
"""),
    md("## Visualise MC results"),
    code("""
top16 = results_df.head(16).copy()
fig = px.bar(
    top16, x="team", y="win_pct",
    color="win_pct", color_continuous_scale="Greens",
    title="2026 WC Champion Probability (Monte Carlo, 10k simulations)",
    labels={"win_pct": "Win Probability (%)"},
)
fig.update_xaxes(tickangle=45)
fig.show()
"""),
    code("""
# Stacked bar: win/final/sf/qf
plot_df = top16.melt(
    id_vars="team",
    value_vars=["win_pct", "final_pct", "sf_pct", "qf_pct"],
    var_name="stage", value_name="probability",
)
fig2 = px.bar(
    plot_df, x="team", y="probability", color="stage",
    title="Stage Reach Probabilities — Top 16 Teams",
    labels={"probability": "Probability (%)"},
    barmode="group",
)
fig2.update_xaxes(tickangle=45)
fig2.show()
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# 06_bracket_simulator.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb06 = nb([
    md("# Notebook 06 — Full Bracket Simulation"),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import config
from src.models import MatchPredictor, MonteCarloSimulator
from src.bracket import BracketSimulator
from src.match_simulator import MatchSimulator
from src.utils import get_logger

log = get_logger("06_bracket")
"""),
    md("## Setup"),
    code("""
model_path = config.OUTPUTS_DIR / "match_predictor.pkl"
try:
    predictor = MatchPredictor.load(model_path)
    print("Loaded trained predictor")
except Exception:
    predictor = None
    print("Using Elo fallback")

mc_sim = MonteCarloSimulator(predictor=predictor, n_simulations=500, random_state=42)
bracket_sim = BracketSimulator(mc_simulator=mc_sim, random_state=42)
"""),
    md("## Run full bracket"),
    code("""
print("Running complete 2026 WC bracket simulation...")
tournament = bracket_sim.run_full_bracket()
print(f"\\n🏆 Champion: {tournament['champion']}")
"""),
    md("## Group Standings"),
    code("""
bracket_sim.print_group_standings()
"""),
    md("## Bracket Tree"),
    code("""
bracket_sim.print_bracket_tree(tournament["bracket_tree"])
"""),
    md("## Golden Boot Leaderboard"),
    code("""
bracket_sim.print_top_scorers(n=20)
"""),
    md("## Team Tournament Statistics"),
    code("""
bracket_sim.print_team_stats()
"""),
    md("## Match-by-match knockout results"),
    code("""
ms = MatchSimulator(predictor=predictor, random_state=42)
for stage_key, stage_label in [("r16","R16"), ("qf","QF"), ("sf","SF"), ("final","FINAL")]:
    matches = tournament["knockout_results"].get(stage_key, [])
    if not matches:
        continue
    print(f"\\n{'='*60}")
    print(f"  {stage_label} RESULTS")
    print(f"{'='*60}")
    for m in matches:
        t1, t2 = m.get("team1","?"), m.get("team2","?")
        s = m.get("score", {})
        g1, g2 = s.get("team1","?"), s.get("team2","?")
        w = m.get("winner","?")
        pen = " (pens)" if m.get("penalties") else ""
        et = " (AET)" if m.get("extra_time") and not m.get("penalties") else ""
        print(f"  {t1} {g1} - {g2} {t2}{et}{pen}  →  Winner: {w}")
"""),
    md("## Winner probability (1000 simulations)"),
    code("""
bracket_sim.print_winner_prediction(n_sims=1000)
"""),
    md("## Export all match results"),
    code("""
all_ko_matches = []
for stage_key in ["r32", "r16", "qf", "sf"]:
    all_ko_matches.extend(tournament["knockout_results"].get(stage_key, []))
if tournament["knockout_results"].get("3rd"):
    all_ko_matches.append(tournament["knockout_results"]["3rd"])
if tournament["knockout_results"].get("final"):
    all_ko_matches.append(tournament["knockout_results"]["final"])

ms_obj = MatchSimulator(predictor=predictor)
df_results = ms_obj.to_dataframe(all_ko_matches)
out_path = config.OUTPUTS_DIR / "bracket_results.csv"
df_results.to_csv(out_path, index=False)
print(f"Exported {len(df_results)} knockout matches → {out_path}")
df_results[["stage","team1","team2","score_t1","score_t2","winner","xg_t1","xg_t2"]].head(20)
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# 07_model_comparison.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb07 = nb([
    md("# Notebook 07 — Model Comparison: ML vs Monte Carlo"),
    code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

import config
from src.models import MatchPredictor, MonteCarloSimulator, _prepare_xy
from src.feature_engineering import build_training_matrix
from src.utils import get_logger

log = get_logger("07_model_comparison")
"""),
    md("## Load data and models"),
    code("""
df = build_training_matrix(min_year=1990, competitive_only=True)
df["date"] = pd.to_datetime(df["date"])

# Backtest splits
val_df  = df[(df["date"].dt.year >= 2015) & (df["date"].dt.year <= 2018)].copy()
test_df = df[df["date"].dt.year >= 2019].copy()

model_path = config.OUTPUTS_DIR / "match_predictor.pkl"
try:
    predictor = MatchPredictor.load(model_path)
    print(f"ML model loaded: {len(predictor.feature_cols)} features")
except Exception as e:
    print(f"Cannot load ML model: {e} — training minimal version")
    train_df = df[df["date"].dt.year <= 2014].copy()
    predictor = MatchPredictor(n_optuna_trials=5)
    predictor.fit(train_df, tune=False)

mc_sim = MonteCarloSimulator(predictor=predictor, n_simulations=500, random_state=42)
"""),
    md("## ML Model Metrics"),
    code("""
def ml_metrics(predictor, df, label):
    X, y_clf, y_g1, y_g2, _ = _prepare_xy(df)
    Xi = predictor.imputer.transform(X)
    Xs = predictor.scaler.transform(Xi)
    proba = predictor.xgb_model.predict_proba(Xs)
    preds = proba.argmax(axis=1)
    ll = log_loss(y_clf, proba)
    acc = accuracy_score(y_clf, preds)
    y_bin = (y_clf == 2).astype(int)
    bs = brier_score_loss(y_bin, proba[:, 2])
    # Score MAE
    l1_pred = np.clip(predictor.poisson_model_t1.predict(Xs), 0, 8)
    l2_pred = np.clip(predictor.poisson_model_t2.predict(Xs), 0, 8)
    mae_g1 = np.abs(y_g1 - l1_pred).mean()
    mae_g2 = np.abs(y_g2 - l2_pred).mean()
    print(f"{label}:")
    print(f"  Log Loss: {ll:.4f} | Accuracy: {acc:.4f} | Brier: {bs:.4f}")
    print(f"  Goals MAE — T1: {mae_g1:.3f}, T2: {mae_g2:.3f}")
    return proba, y_clf, y_g1, y_g2

val_proba, val_y, val_g1, val_g2 = ml_metrics(predictor, val_df, "Validation (2015-2018)")
test_proba, test_y, test_g1, test_g2 = ml_metrics(predictor, test_df, "Test (2019+)")
"""),
    md("## MC-implied probabilities vs actual outcomes"),
    code("""
# Generate MC predictions on validation set
sample_val = val_df.sample(min(100, len(val_df)), random_state=42)
mc_preds = []
for _, row in sample_val.iterrows():
    team1, team2 = row["team1"], row["team2"]
    prob = mc_sim._mc_win_prob(team1, team2, "MetLife Stadium", "group", n=200)
    mc_preds.append({
        "team1": team1, "team2": team2,
        "mc_win_t1": prob["team1"], "mc_win_t2": prob["team2"],
        "actual_result": row["result"],
    })
mc_df = pd.DataFrame(mc_preds)
mc_correct = ((mc_df["mc_win_t1"] > 0.5) == (mc_df["actual_result"] == 1)).mean()
print(f"MC directional accuracy (win/loss only): {mc_correct:.3f}")
"""),
    md("## Calibration comparison"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, proba, y, label in [
    (axes[0], val_proba, val_y, "Validation"),
    (axes[1], test_proba, test_y, "Test"),
]:
    y_bin = (y == 2).astype(int)
    frac, mean_pred = calibration_curve(y_bin, proba[:, 2], n_bins=8)
    ax.plot(mean_pred, frac, "o-", label="ML Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title(f"Calibration — {label}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
plt.tight_layout()
plt.show()
"""),
    md("## Ensemble analysis: optimal ML vs MC weighting"),
    code("""
alpha_range = np.linspace(0, 1, 11)
best_alpha, best_ll = 0.5, float("inf")

X_val, y_val, _, _, _ = _prepare_xy(val_df)
Xi = predictor.imputer.transform(X_val)
Xs = predictor.scaler.transform(Xi)
ml_val_proba = predictor.xgb_model.predict_proba(Xs)

mc_win_pct = mc_df["mc_win_t1"].values[:len(ml_val_proba)]
mc_proba_approx = np.column_stack([
    1 - mc_win_pct - 0.1,
    np.full(len(mc_win_pct), 0.1),
    mc_win_pct,
]).clip(0, 1)
mc_proba_approx /= mc_proba_approx.sum(axis=1, keepdims=True)
n_compare = min(len(ml_val_proba), len(mc_proba_approx))

logloss_by_alpha = []
for alpha in alpha_range:
    combined = alpha * ml_val_proba[:n_compare] + (1 - alpha) * mc_proba_approx[:n_compare]
    ll = log_loss(y_val[:n_compare], combined)
    logloss_by_alpha.append(ll)
    if ll < best_ll:
        best_ll = ll
        best_alpha = alpha

plt.figure(figsize=(9, 4))
plt.plot(alpha_range, logloss_by_alpha, "o-", color="steelblue")
plt.axvline(best_alpha, color="red", linestyle="--", label=f"Optimal α={best_alpha:.1f}")
plt.xlabel("α (weight on ML model)")
plt.ylabel("Validation Log Loss")
plt.title("Ensemble: ML + MC Log Loss by Mixing Weight")
plt.legend(); plt.tight_layout(); plt.show()
print(f"Optimal α (ML weight): {best_alpha:.2f} → Log Loss: {best_ll:.4f}")
"""),
    md("## 2026 Predictions Comparison Table"),
    code("""
# Load MC results if available
mc_path = config.OUTPUTS_DIR / "mc_tournament_results.csv"
try:
    mc_results = pd.read_csv(mc_path)
    mc_dict = dict(zip(mc_results["team"], mc_results["win_pct"]))
except Exception:
    mc_dict = {t: 1/48 for t in config.ALL_TEAMS}

# ML win probabilities via feature vectors
from src.feature_engineering import build_match_features
ml_wins = {}
for team in config.ALL_TEAMS:
    try:
        fv = build_match_features(team, "Germany", "2026-07-01", "MetLife Stadium", "final")
        p = predictor.predict_proba(fv)
        ml_wins[team] = p["win_t1"]
    except Exception:
        ml_wins[team] = 1 / 48

comparison_df = pd.DataFrame({
    "team": config.ALL_TEAMS,
    "ml_win_pct": [round(ml_wins.get(t, 0) * 100, 1) for t in config.ALL_TEAMS],
    "mc_win_pct": [round(mc_dict.get(t, 1/48) * 100, 1) for t in config.ALL_TEAMS],
    "elo_rating": [config.INITIAL_ELO.get(t, 1500) for t in config.ALL_TEAMS],
    "fifa_ranking": [config.FIFA_RANKINGS.get(t, 48) for t in config.ALL_TEAMS],
})
comparison_df["ensemble_win_pct"] = (
    best_alpha * comparison_df["ml_win_pct"] + (1 - best_alpha) * comparison_df["mc_win_pct"]
).round(1)
comparison_df = comparison_df.sort_values("ensemble_win_pct", ascending=False).reset_index(drop=True)

print("\\n2026 WC Predictions Comparison Table:")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(config.OUTPUTS_DIR / "predictions_comparison.csv", index=False)
print(f"\\nSaved → {config.OUTPUTS_DIR}/predictions_comparison.csv")
"""),
    md("## Final visualisation: win probability distribution"),
    code("""
top20 = comparison_df.head(20)
fig = go.Figure()
fig.add_trace(go.Bar(name="ML", x=top20["team"], y=top20["ml_win_pct"], marker_color="steelblue"))
fig.add_trace(go.Bar(name="MC", x=top20["team"], y=top20["mc_win_pct"], marker_color="coral"))
fig.add_trace(go.Bar(name="Ensemble", x=top20["team"], y=top20["ensemble_win_pct"], marker_color="green"))
fig.update_layout(
    barmode="group",
    title="2026 WC Champion Probability — ML vs MC vs Ensemble (Top 20 Teams)",
    xaxis_title="Team", yaxis_title="Win Probability (%)",
    xaxis_tickangle=45,
)
fig.show()
"""),
])

# ── Write all notebooks ───────────────────────────────────────────────────────
notebooks = {
    "01_data_collection.ipynb": nb01,
    "02_eda.ipynb": nb02,
    "03_feature_engineering.ipynb": nb03,
    "04_ml_model.ipynb": nb04,
    "05_monte_carlo.ipynb": nb05,
    "06_bracket_simulator.ipynb": nb06,
    "07_model_comparison.ipynb": nb07,
}

for fname, notebook in notebooks.items():
    path = NB_DIR / fname
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print(f"Created: {path}")

print("\nAll 7 notebooks created successfully.")
