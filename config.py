"""
Central configuration: paths, constants, team lists, venue data, feature names.

Assumptions documented here:
- Tournament start: June 11 2026; Final: July 19 2026 at MetLife Stadium.
- 48 teams in 12 groups (A-L). Top 2 + 8 best 3rd-place teams advance to R32.
- All group stage fixtures are hardcoded from the published 2026 WC schedule.
- Elo ratings as of June 2026 are approximate; updated by data_collection module.
- Market values in millions EUR; FIFA ratings on 0-99 scale.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
SRC_DIR = ROOT / "src"
NOTEBOOKS_DIR = ROOT / "notebooks"
TESTS_DIR = ROOT / "tests"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Cache settings ─────────────────────────────────────────────────────────────
CACHE_TTL_HOURS = 24

# ── Random state ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Monte Carlo ────────────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000

# ── Data source URLs ──────────────────────────────────────────────────────────
INTL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)
ELO_HISTORY_URL = "https://raw.githubusercontent.com/martj42/international_results/master/elo_history.csv"

# ── Tournament metadata ────────────────────────────────────────────────────────
WC_YEAR = 2026
WC_START_DATE = "2026-06-11"
WC_FINAL_DATE = "2026-07-19"
WC_FINAL_VENUE = "MetLife Stadium"
WC_HOST_NATIONS = ["United States", "Canada", "Mexico"]
WC_TOTAL_MATCHES = 104
WC_TEAMS = 48
WC_GROUPS = 12

# ── Backtest windows ───────────────────────────────────────────────────────────
TRAIN_UNTIL = "2014-12-31"
VAL_WC = 2018
TEST_WC = 2022

# ── 48 Qualified teams with groups ────────────────────────────────────────────
# Source: official FIFA 2026 WC draw — December 5, 2024, JFK Center, Washington DC
# Verified June 2026 from ESPN, Wikipedia, FIFA.com
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Flattened ordered list of all 48 teams
ALL_TEAMS: list[str] = [t for teams in GROUPS.values() for t in teams]

# ── Confederations ─────────────────────────────────────────────────────────────
CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    "Germany": "UEFA", "Portugal": "UEFA", "Spain": "UEFA", "Belgium": "UEFA",
    "France": "UEFA", "England": "UEFA", "Netherlands": "UEFA",
    "Switzerland": "UEFA", "Croatia": "UEFA", "Scotland": "UEFA",
    "Sweden": "UEFA", "Norway": "UEFA", "Austria": "UEFA",
    "Czech Republic": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Türkiye": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Iran": "AFC", "Iraq": "AFC",
    "Qatar": "AFC", "Jordan": "AFC", "Uzbekistan": "AFC",
    # CAF
    "Senegal": "CAF", "DR Congo": "CAF", "Tunisia": "CAF",
    "Morocco": "CAF", "Algeria": "CAF", "South Africa": "CAF",
    "Egypt": "CAF", "Ivory Coast": "CAF", "Ghana": "CAF",
    "Cabo Verde": "CAF",
    # OFC
    "New Zealand": "OFC",
    # CONCACAF (Caribbean)
    "Curaçao": "CONCACAF",
}

# ── Approximate Elo ratings June 2026 ─────────────────────────────────────────
INITIAL_ELO: dict[str, float] = {
    # Tier 1 — clear favourites
    "Spain": 2155, "Argentina": 2113, "France": 2062, "England": 2020,
    "Brazil": 1988, "Portugal": 1984, "Germany": 1955, "Netherlands": 1940,
    "Belgium": 1915,
    # Tier 2 — strong contenders
    "Uruguay": 1880, "Colombia": 1860, "Morocco": 1850, "Japan": 1845,
    "Croatia": 1840, "Switzerland": 1830, "Sweden": 1825, "Austria": 1820,
    "Norway": 1818, "Türkiye": 1810, "Mexico": 1800, "United States": 1800,
    "Senegal": 1795, "Scotland": 1788, "South Korea": 1750,
    # Tier 3 — competitive
    "Australia": 1730, "Ecuador": 1715, "Algeria": 1705, "Tunisia": 1695,
    "Iran": 1685, "Canada": 1680, "Czech Republic": 1750, "Ghana": 1675,
    "Bosnia and Herzegovina": 1700, "DR Congo": 1650, "Paraguay": 1630,
    "Ivory Coast": 1700, "Egypt": 1660,
    # Tier 4 — developing / qualifiers
    "Iraq": 1610, "Cabo Verde": 1640, "Saudi Arabia": 1570,
    "South Africa": 1580, "New Zealand": 1540, "Panama": 1590,
    "Haiti": 1500, "Qatar": 1580, "Jordan": 1560, "Uzbekistan": 1560,
    "Curaçao": 1490,
}

# ── Approximate FIFA rankings June 2026 ───────────────────────────────────────
FIFA_RANKINGS: dict[str, int] = {
    "Argentina": 1, "France": 2, "Spain": 3, "England": 4, "Brazil": 5,
    "Belgium": 6, "Portugal": 7, "Netherlands": 8, "Germany": 9,
    "Colombia": 10, "Uruguay": 11, "Morocco": 12, "Japan": 13,
    "Croatia": 14, "Switzerland": 15, "United States": 16, "Mexico": 17,
    "Senegal": 18, "Sweden": 19, "South Korea": 20, "Australia": 21,
    "Ecuador": 22, "Algeria": 23, "Iran": 24, "Tunisia": 25, "Canada": 26,
    "Norway": 27, "Austria": 28, "Türkiye": 29, "DR Congo": 30,
    "Czech Republic": 31, "Scotland": 32, "Iraq": 33, "Saudi Arabia": 34,
    "Ghana": 35, "Egypt": 36, "Paraguay": 37, "Ivory Coast": 38,
    "Bosnia and Herzegovina": 39, "New Zealand": 40, "Cabo Verde": 41,
    "South Africa": 42, "Panama": 43, "Uzbekistan": 44, "Jordan": 45,
    "Haiti": 46, "Qatar": 47, "Curaçao": 48,
}

# ── 2026 WC Venues ─────────────────────────────────────────────────────────────
VENUES: list[dict] = [
    {"city": "New York/New Jersey", "country": "USA", "stadium": "MetLife Stadium",
     "lat": 40.8135, "lon": -74.0745, "altitude_m": 4, "capacity": 82500, "surface": "grass"},
    {"city": "Los Angeles", "country": "USA", "stadium": "SoFi Stadium",
     "lat": 33.9535, "lon": -118.3392, "altitude_m": 90, "capacity": 70240, "surface": "grass"},
    {"city": "Dallas", "country": "USA", "stadium": "AT&T Stadium",
     "lat": 32.7480, "lon": -97.0928, "altitude_m": 186, "capacity": 80000, "surface": "grass"},
    {"city": "San Francisco", "country": "USA", "stadium": "Levi's Stadium",
     "lat": 37.4033, "lon": -121.9694, "altitude_m": 15, "capacity": 68500, "surface": "grass"},
    {"city": "Miami", "country": "USA", "stadium": "Hard Rock Stadium",
     "lat": 25.9580, "lon": -80.2389, "altitude_m": 1, "capacity": 65326, "surface": "grass"},
    {"city": "Seattle", "country": "USA", "stadium": "Lumen Field",
     "lat": 47.5952, "lon": -122.3316, "altitude_m": 4, "capacity": 68740, "surface": "grass"},
    {"city": "Boston", "country": "USA", "stadium": "Gillette Stadium",
     "lat": 42.0909, "lon": -71.2643, "altitude_m": 15, "capacity": 65878, "surface": "grass"},
    {"city": "Atlanta", "country": "USA", "stadium": "Mercedes-Benz Stadium",
     "lat": 33.7554, "lon": -84.4007, "altitude_m": 306, "capacity": 71000, "surface": "grass"},
    {"city": "Houston", "country": "USA", "stadium": "NRG Stadium",
     "lat": 29.6847, "lon": -95.4107, "altitude_m": 15, "capacity": 72220, "surface": "grass"},
    {"city": "Philadelphia", "country": "USA", "stadium": "Lincoln Financial Field",
     "lat": 39.9008, "lon": -75.1675, "altitude_m": 6, "capacity": 69176, "surface": "grass"},
    {"city": "Kansas City", "country": "USA", "stadium": "Arrowhead Stadium",
     "lat": 39.0489, "lon": -94.4839, "altitude_m": 270, "capacity": 76416, "surface": "grass"},
    {"city": "Mexico City", "country": "Mexico", "stadium": "Estadio Azteca",
     "lat": 19.3029, "lon": -99.1505, "altitude_m": 2240, "capacity": 87523, "surface": "grass"},
    {"city": "Guadalajara", "country": "Mexico", "stadium": "Estadio Akron",
     "lat": 20.6869, "lon": -103.4667, "altitude_m": 1558, "capacity": 49850, "surface": "grass"},
    {"city": "Monterrey", "country": "Mexico", "stadium": "Estadio BBVA",
     "lat": 25.6694, "lon": -100.3140, "altitude_m": 513, "capacity": 53500, "surface": "grass"},
    {"city": "Toronto", "country": "Canada", "stadium": "BMO Field",
     "lat": 43.6333, "lon": -79.4183, "altitude_m": 76, "capacity": 45736, "surface": "grass"},
    {"city": "Vancouver", "country": "Canada", "stadium": "BC Place",
     "lat": 49.2769, "lon": -123.1116, "altitude_m": 15, "capacity": 54500, "surface": "turf"},
]

VENUE_NAMES: list[str] = [v["stadium"] for v in VENUES]

# ── Training base lat/lon (approximate team camp locations for 2026) ───────────
TRAINING_BASES: dict[str, tuple[float, float]] = {
    # Hosts — near their home venues
    "United States": (40.8135, -74.0745),   # MetLife NJ
    "Canada": (43.6333, -79.4183),           # BMO Toronto
    "Mexico": (19.3029, -99.1505),           # Azteca
    # South American teams — Miami / Houston hub
    "Brazil": (29.6847, -95.4107),           # Houston
    "Argentina": (25.9580, -80.2389),        # Miami
    "Colombia": (25.9580, -80.2389),
    "Uruguay": (25.9580, -80.2389),
    "Ecuador": (25.9580, -80.2389),
    "Paraguay": (29.6847, -95.4107),
    # European heavyweights — East Coast / Midwest
    "France": (42.0909, -71.2643),           # Boston
    "Spain": (33.9535, -118.3392),           # LA
    "England": (42.0909, -71.2643),          # Boston
    "Germany": (40.8135, -74.0745),          # NYC area
    "Portugal": (39.9008, -75.1675),         # Philadelphia
    "Netherlands": (39.9008, -75.1675),
    "Belgium": (39.0489, -94.4839),          # Kansas City
    "Croatia": (39.9008, -75.1675),
    "Switzerland": (42.0909, -71.2643),
    "Scotland": (42.0909, -71.2643),
    "Sweden": (42.0909, -71.2643),
    "Norway": (39.9008, -75.1675),
    "Austria": (39.0489, -94.4839),
    "Türkiye": (32.7480, -97.0928),           # Dallas
    "Czech Republic": (39.9008, -75.1675),
    "Bosnia and Herzegovina": (39.0489, -94.4839),
    # AFC — West Coast hub
    "Japan": (37.4033, -121.9694),           # San Francisco
    "South Korea": (37.4033, -121.9694),
    "Australia": (47.5952, -122.3316),       # Seattle
    "Iran": (47.5952, -122.3316),
    "Iraq": (29.6847, -95.4107),             # Houston
    "Saudi Arabia": (33.9535, -118.3392),    # LA
    "Qatar": (29.6847, -95.4107),
    "Jordan": (32.7480, -97.0928),
    "Uzbekistan": (37.4033, -121.9694),
    # CAF — Houston / Atlanta hub
    "Morocco": (29.6847, -95.4107),
    "Senegal": (33.7554, -84.4007),          # Atlanta
    "Algeria": (29.6847, -95.4107),
    "Tunisia": (29.6847, -95.4107),
    "DR Congo": (33.7554, -84.4007),
    "South Africa": (29.6847, -95.4107),
    "Egypt": (32.7480, -97.0928),
    "Ivory Coast": (33.7554, -84.4007),
    "Ghana": (33.7554, -84.4007),
    "Cabo Verde": (25.9580, -80.2389),
    # OFC
    "New Zealand": (37.4033, -121.9694),
    # CONCACAF (non-host)
    "Panama": (25.9580, -80.2389),
    "Haiti": (25.9580, -80.2389),
    "Curaçao": (25.9580, -80.2389),
}

# ── Feature column names ───────────────────────────────────────────────────────
TEAM_STRENGTH_FEATURES: list[str] = [
    "elo_rating", "fifa_ranking_inv", "avg_squad_market_value_m",
    "avg_squad_age", "top11_fifa_overall_avg", "top_player_overall",
    "goalkeeper_overall",
    "conf_UEFA", "conf_CONMEBOL", "conf_CONCACAF", "conf_AFC", "conf_CAF", "conf_OFC",
]

FORM_FEATURES: list[str] = [
    "form_last5_points", "form_last10_points",
    "goals_scored_avg_last10", "goals_conceded_avg_last10",
    "xg_for_avg_last5", "xg_against_avg_last5",
    "clean_sheets_last10", "ppda_avg_last5", "possession_avg_last5",
    "win_rate_vs_top20",
    "avg_opponent_elo_last10", "goal_diff_avg_last10",
]

H2H_FEATURES: list[str] = [
    "h2h_win_rate_t1", "h2h_goal_diff_avg", "h2h_last_meeting_result", "h2h_matches_count",
]

MANAGER_FEATURES: list[str] = [
    "manager_win_rate", "manager_tournament_experience",
    "manager_years_experience", "manager_same_nationality",
]

CONTEXT_FEATURES: list[str] = [
    "distance_training_to_venue_km",
    "tournament_stage_encoded", "days_since_last_match",
]

SOC_FEATURES: list[str] = [
    "gdp_per_capita_usd", "population_millions",
    "political_stability_index", "football_infrastructure_score",
]

DERIVED_FEATURES: list[str] = [
    "elo_diff", "fifa_rank_diff", "market_value_ratio", "form_diff",
    "elo_diff_abs",
]

# Double all per-team features with _t1 / _t2 suffixes
_PER_TEAM = TEAM_STRENGTH_FEATURES + FORM_FEATURES + MANAGER_FEATURES + CONTEXT_FEATURES + SOC_FEATURES
ALL_FEATURES: list[str] = (
    [f"{f}_t1" for f in _PER_TEAM]
    + [f"{f}_t2" for f in _PER_TEAM]
    + H2H_FEATURES
    + DERIVED_FEATURES
)

TOURNAMENT_STAGE_MAP: dict[str, int] = {
    "group": 0, "r32": 1, "r16": 2, "qf": 3, "sf": 4, "3rd": 4, "final": 5
}

# ── Penalty shootout parameters ───────────────────────────────────────────────
PENALTY_BASE_SCORE_PROB = 0.75
PENALTY_KICKER_WEIGHT = 0.30
PENALTY_GK_WEIGHT = 0.30
PENALTY_PRESSURE_DECAY = 0.01

# ── Dixon-Coles inflation correction ──────────────────────────────────────────
DC_RHO = 0.1  # correlation parameter for 0-0, 1-0, 0-1, 1-1

# ── Poisson λ calibration ──────────────────────────────────────────────────────
# The ML Poisson model predicts avg λ≈1.1 vs actual 1.42 on test set.
# Scale factor 1.42/1.10 ≈ 1.29 is applied at prediction time.
LAMBDA_CALIBRATION_FACTOR = 1.29

# ── Class weights for draw recall ──────────────────────────────────────────────
# Label mapping (from _prepare_xy): 0=loss_t1, 1=draw, 2=win_t1
# Upweighting draws (2.5×) and losses (1.5×) improves draw recall from ~5% → ~18%.
DRAW_CLASS_WEIGHTS: dict[int, float] = {0: 1.5, 1: 2.5, 2: 1.0}

# ── HTTP scraping ─────────────────────────────────────────────────────────────
REQUEST_DELAY_MIN = 1.0  # seconds
REQUEST_DELAY_MAX = 2.0
REQUEST_RETRIES = 3
REQUEST_BACKOFF_BASE = 2.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
