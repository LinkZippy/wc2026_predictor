#!/usr/bin/env python3
"""
Flask API wrapper for WC 2026 Match Predictor.

Install dependencies (once):
    pip install flask flask-cors

Run the server:
    python api.py

The trained model must exist at:
    data/outputs/match_predictor.pkl
    (run retrain_and_simulate.py if it doesn't exist yet)
"""

import sys
import warnings
import logging
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from scipy.stats import poisson as _poisson_rv

import config
from src.match_simulator import MatchSimulator

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allows all origins so GitHub Pages can call this server

# ── Model singleton ───────────────────────────────────────────────────────────
_predictor = None


def _load_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    pkl = ROOT / "data" / "outputs" / "match_predictor.pkl"
    if pkl.exists():
        _predictor = joblib.load(pkl)
        logging.warning("Model loaded from %s", pkl)
    else:
        logging.warning("model not found at %s — using Elo fallback", pkl)
    return _predictor


# ── Helpers ───────────────────────────────────────────────────────────────────
def _serialize(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _dc_probs(l1: float, l2: float) -> dict:
    """Recompute W/D/L probabilities from adjusted xG lambdas (Dixon-Coles)."""
    rho = config.DC_RHO
    n = 9
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p = float(_poisson_rv.pmf(i, l1) * _poisson_rv.pmf(j, l2))
            if i == 0 and j == 0:
                tau = 1 - l1 * l2 * rho
            elif i == 1 and j == 0:
                tau = 1 + l2 * rho
            elif i == 0 and j == 1:
                tau = 1 + l1 * rho
            elif i == 1 and j == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            mat[i, j] = max(0.0, p * tau)
    total = mat.sum() or 1.0
    mat /= total
    win1 = float(np.tril(mat, -1).sum())
    draw = float(np.trace(mat))
    win2 = float(np.triu(mat, 1).sum())
    t = win1 + draw + win2 or 1.0
    return {"win_t1": win1 / t, "draw": draw / t, "win_t2": win2 / t}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _load_predictor() is not None})


@app.route("/teams", methods=["GET"])
def get_teams():
    return jsonify(sorted(config.ALL_TEAMS))


@app.route("/venues", methods=["GET"])
def get_venues():
    return jsonify(config.VENUE_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        team1 = str(payload["team1"])
        team2 = str(payload["team2"])
        date  = str(payload.get("date",  "2026-06-20"))
        venue = str(payload.get("venue", "MetLife Stadium"))
        stage = str(payload.get("stage", "group"))

        # ── Slider values from the frontend ──────────────────────────────────
        # weather:   0 (severe) → 100 (ideal)
        # form_a/b:  -10 (poor form) → +10 (hot streak), as percentages
        # stakes:    0 (friendly) → 100 (World Cup Final)
        # sim_depth: Monte Carlo iteration count (100–5000)
        weather   = float(payload.get("weather",    50))
        form_a    = float(payload.get("form_a",      0))
        form_b    = float(payload.get("form_b",      0))
        stakes    = float(payload.get("stakes",     50))
        sim_depth = int(payload.get("sim_depth",   500))

        # ── Run base simulation (existing code, untouched) ────────────────────
        sim = MatchSimulator(
            predictor=_load_predictor(),
            n_mc=min(max(sim_depth, 100), 5000),
        )
        result = sim.simulate(team1, team2, date=date, venue=venue, stage=stage)

        # ── Apply slider adjustments to xG, then recompute probabilities ──────
        #
        # weather  : 0→×0.75   50→×1.00   100→×1.15  (poor weather = fewer goals)
        # form_a/b : -10%→×0.90   0→×1.00   +10%→×1.10
        # stakes   : 0→×1.00   100→×0.85               (big game = more cautious)
        #
        weather_factor = 0.75 + (weather / 100.0) * 0.40   # [0.75, 1.15]
        form_a_factor  = 1.0  + (form_a  / 100.0)           # [0.90, 1.10]
        form_b_factor  = 1.0  + (form_b  / 100.0)           # [0.90, 1.10]
        stakes_factor  = 1.0  - (stakes  / 100.0) * 0.15   # [1.00, 0.85]

        xg1_base = float(result["xg"]["team1"])
        xg2_base = float(result["xg"]["team2"])
        adj_xg1 = max(0.1, xg1_base * weather_factor * form_a_factor * stakes_factor)
        adj_xg2 = max(0.1, xg2_base * weather_factor * form_b_factor * stakes_factor)

        result["xg"]["team1"]  = round(adj_xg1, 3)
        result["xg"]["team2"]  = round(adj_xg2, 3)
        result["ml_win_prob"]  = _dc_probs(adj_xg1, adj_xg2)
        result["n_mc"]         = sim_depth
        result["adjustments"]  = {
            "weather_factor": round(weather_factor, 3),
            "form_a_factor":  round(form_a_factor,  3),
            "form_b_factor":  round(form_b_factor,  3),
            "stakes_factor":  round(stakes_factor,  3),
        }

        return jsonify(_serialize(result))

    except KeyError as exc:
        return jsonify({"error": f"Missing required field: {exc}"}), 400
    except Exception as exc:
        logging.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _load_predictor()
    app.run(host="0.0.0.0", port=5001, debug=False)
