#!/usr/bin/env python3
"""
WC 2026 match predictor — command-line interface.

Usage
-----
# Predict a single match:
    python3 predict.py "United States" "Paraguay"

# Specify venue / stage:
    python3 predict.py "Brazil" "France" --venue "MetLife Stadium" --stage sf

# Run all fixtures for a date:
    python3 predict.py --date 2026-06-13

# Run today's fixtures:
    python3 predict.py --today
"""

import argparse
import sys
import warnings
import logging
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import joblib
from src.match_simulator import MatchSimulator, format_match_output, save_match_report
from src.live_data import scrape_live_lineup, run_date_prediction


def load_predictor():
    pkl = ROOT / "data" / "outputs" / "match_predictor.pkl"
    if not pkl.exists():
        print(f"[error] Model not found at {pkl}")
        sys.exit(1)
    return joblib.load(pkl)


def predict_match(t1: str, t2: str, match_date: str, venue: str, stage: str):
    predictor = load_predictor()
    sim = MatchSimulator(predictor=predictor)

    try:
        d = datetime.strptime(match_date, "%Y-%m-%d").date()
    except ValueError:
        d = date.today()

    lineup = scrape_live_lineup(t1, t2, d)
    result = sim.simulate(t1, t2, date=match_date, venue=venue, stage=stage)
    output = format_match_output(result, lineup=lineup)
    print(output)

    safe = lambda s: s.replace(" ", "_")
    report_path = ROOT / "data" / "outputs" / "match_reports" / f"{match_date}_{safe(t1)}_vs_{safe(t2)}.txt"
    save_match_report(result, report_path, lineup=lineup)
    print(f"\n[saved] {report_path}")


def predict_date(match_date: str):
    results = run_date_prediction(match_date)
    if not results:
        print(f"No fixtures found for {match_date}.")
        return
    for r in results:
        print(r["formatted_output"])
        print()


def main():
    parser = argparse.ArgumentParser(
        description="WC 2026 match predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("teams", nargs="*", metavar="TEAM",
                        help='Two team names, e.g. "United States" "Paraguay"')
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Match date YYYY-MM-DD (default: today)")
    parser.add_argument("--today", action="store_true",
                        help="Run all fixtures scheduled for today")
    parser.add_argument("--venue", default="Neutral",
                        help="Stadium name (default: Neutral)")
    parser.add_argument("--stage", default="group",
                        choices=["group", "r32", "r16", "qf", "sf", "final"],
                        help="Tournament stage (default: group)")

    args = parser.parse_args()

    if args.today:
        predict_date(date.today().isoformat())
    elif not args.teams:
        predict_date(args.date)
    elif len(args.teams) == 2:
        predict_match(args.teams[0], args.teams[1], args.date, args.venue, args.stage)
    else:
        parser.error("Provide exactly two team names, or use --date / --today")


if __name__ == "__main__":
    main()
