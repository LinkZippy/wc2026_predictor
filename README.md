# 2026 FIFA World Cup Prediction System

End-to-end ML + Monte Carlo bracket prediction engine for the 2026 FIFA World Cup.  
Covers all 48 teams, 104 matches, 10 data sources, XGBoost/LightGBM/CatBoost ensemble, Dixon-Coles Poisson score model, and 10,000-simulation Monte Carlo bracket.

---

## Project Structure

```
wc2026_predictor/
├── config.py                      ← all constants, paths, team lists, venue data
├── requirements.txt
├── data/
│   ├── raw/                       ← downloaded data (auto-populated)
│   ├── processed/                 ← feature matrices, H2H stats
│   └── outputs/                   ← predictions, bracket results, run.log
├── src/
│   ├── data_collection.py         ← 10 data source downloaders
│   ├── feature_engineering.py     ← build_match_features() + training matrix
│   ├── models.py                  ← MatchPredictor + MonteCarloSimulator
│   ├── bracket.py                 ← BracketSimulator engine
│   ├── match_simulator.py         ← MatchSimulator (per-match output)
│   └── utils.py                   ← logging, caching, retry decorator
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_model.ipynb
│   ├── 05_monte_carlo.ipynb
│   ├── 06_bracket_simulator.ipynb
│   └── 07_model_comparison.ipynb
└── tests/
    ├── test_bracket.py
    ├── test_feature_engineering.py
    ├── test_penalty_shootout.py
    └── test_poisson.py
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests (all should pass)
pytest tests/ -v

# 3. Open notebooks in order
jupyter lab notebooks/
```

---

## Notebooks Walkthrough

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Collection | Download all 10 sources, cache to `data/raw/` |
| 02 | EDA | Distribution plots, Elo vs results, group compositions |
| 03 | Feature Engineering | Build feature vectors, training matrix, correlation analysis |
| 04 | ML Model | XGBoost + LightGBM + CatBoost ensemble, Optuna tuning, SHAP |
| 05 | Monte Carlo | 10,000 full tournament simulations, penalty model |
| 06 | Bracket Simulator | Run full bracket, print standings + bracket tree + top scorers |
| 07 | Model Comparison | ML vs MC metrics, calibration, ensemble weighting, 2026 predictions |

---

## Data Sources

| # | Source | Method | Fallback |
|---|--------|--------|---------|
| 1 | International results (martj42/GitHub) | Direct CSV download | Synthetic (same schema) |
| 2 | Elo ratings (eloratings.net) | HTTP scrape | `config.INITIAL_ELO` |
| 3 | FIFA rankings history (Kaggle) | Local CSV or config defaults | Hardcoded FIFA_RANKINGS |
| 4A | Transfermarkt squad values | HTTP scrape | Synthetic market values |
| 4B | EA FC 24/25 player ratings (Kaggle) | Local CSV or synthetic | Synthetic per-team |
| 5 | StatsBomb open data | `statsbombpy` library | Synthetic xG/PPDA |
| 6 | Manager data (Transfermarkt) | Scrape + hardcoded known coaches | Synthetic |
| 7 | World Bank socioeconomic | `wbdata` library | Synthetic |
| 8 | Venue & geography | Hardcoded 16 venues | — |
| 9 | Head-to-head history | Computed from source 1 | — |
| 10 | 2026 WC fixtures | Hardcoded (72 group stage) | — |

**All downloaders**: rate-limited (1-2s), 3 retries with exponential backoff, 24-hour cache TTL, graceful synthetic fallback.

---

## Model Architecture

### MatchPredictor (ML)

- **Preprocessing**: `SimpleImputer(median)` + `RobustScaler` + missing-indicator flags
- **Outcome model**: 3-class (W/D/L) ensemble of XGBoost + LightGBM + CatBoost  
  - Hyperparameter tuning via Optuna (50 trials, TPE sampler)
- **Score model**: Two Poisson regressors (one per team) with Dixon-Coles low-score correction
- **Ensemble**: `0.4 × XGB + 0.35 × LGBM + 0.25 × CatBoost`, then `0.7 × classifier + 0.3 × Poisson`

### MonteCarloSimulator

- Samples match scores from Poisson(λ₁, λ₂)
- Handles extra time and penalty shootouts
- Penalty model: `P(score) = 0.75 + kicker_quality × 0.3 − GK_quality × 0.3 − pressure × 0.01`
- Distributes goals to players weighted by `overall²`
- Runs 10,000 complete tournaments; outputs W/Final/SF/QF/R16 probabilities per team

### Feature Vector (78 features)

Per-team (×2) features: Elo, FIFA rank (inverted), squad market value, squad age, FIFA overall (top11 + top1 + GK), confederation one-hot, form (last 5/10), xG, PPDA, possession, clean sheets, win rate vs top-20, manager stats, travel distance, stage encoding, GDP per capita, population, political stability, infrastructure score.

Derived: elo_diff, fifa_rank_diff, market_value_ratio, form_diff.

H2H: win rate, goal diff average, last meeting result, match count.

---

## Backtesting

| Split | Period | Use |
|-------|--------|-----|
| Train | 1990–2014 | Model fitting |
| Validation | 2015–2018 | Optuna tuning, ensemble weighting |
| Test | 2019+ | Held-out evaluation |

Reported metrics: log loss, W/D/L accuracy, Brier score, goals MAE, calibration curves.

---

## 2026 WC Context

- **Tournament dates**: 11 June – 19 July 2026
- **Teams**: 48 (12 groups of 4)
- **Hosts**: USA, Canada, Mexico (auto-qualified)
- **Venues**: 16 stadiums across USA/Canada/Mexico (see `config.VENUES`)
- **Qualification**: Top 2 per group + 8 best 3rd-place → R32 (32 teams)
- **Final**: MetLife Stadium, East Rutherford NJ, 19 July 2026
- **Tiebreakers**: Points → GD → GF → H2H pts → H2H GD → H2H GF → Fair play → Draw

---

## Assumptions

1. Group assignments use the December 2024 official draw.  Groups H placeholder teams may differ from final draw — update `config.GROUPS` to correct.
2. June 2026 Elo ratings are approximate; run notebook 01 to pull live values.
3. Kaggle datasets (FIFA player ratings, FIFA rankings history) must be downloaded manually and placed in `data/raw/` as `fc24_players.csv` and `fifa_world_ranking.csv`. If absent, synthetic data is used automatically.
4. StatsBomb free tier may not cover all international competitions; missing teams receive synthetic averages.
5. Transfermarkt scraping is subject to rate limits and HTML changes; synthetic fallback activates on failure.
6. Travel distances use great-circle approximation; actual travel routes differ.
7. Manager data: 8 high-profile coaches are hardcoded; all others are synthetic.
8. The R32 bracket seeding follows sequential pairing (1st vs 2nd in adjacent groups) — the exact FIFA 2026 seeding grid for third-place teams will be confirmed after the group stage.

---

## Running Tests

```bash
pytest tests/ -v
# Expected: 28 passed
```

---

## Outputs

After running all notebooks:

| File | Description |
|------|-------------|
| `data/outputs/match_predictor.pkl` | Serialised MatchPredictor |
| `data/outputs/mc_tournament_results.csv` | MC probabilities for all 48 teams |
| `data/outputs/bracket_results.csv` | All knockout match results |
| `data/outputs/predictions_comparison.csv` | ML vs MC vs ensemble comparison table |
| `data/outputs/run.log` | Full execution log |
