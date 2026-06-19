#!/usr/bin/env python3
"""
Auto-updater for results.csv and wc2026_fixtures.csv.

What it does
------------
1. results.csv      — Re-downloads from martj42/international_results on GitHub,
                      then supplements with live WC 2026 scores scraped from the
                      ESPN public API for any dates the GitHub repo hasn't caught up on.
2. wc2026_fixtures  — Patches in actual scores (score_t1, score_t2, status) for
                      every past group-stage match by querying ESPN.
3. Cache busting    — Deletes the .meta sidecar files for h2h_stats and
                      training_matrix so they are rebuilt on the next pipeline run.
4. Elo ratings      — Force-refreshes elo_ratings.csv from eloratings.net.

Usage
-----
    python3 auto_update.py                  # run once
    python3 auto_update.py --watch          # loop every 24 hours
    python3 auto_update.py --interval 6     # loop every 6 hours
    python3 auto_update.py --force          # bypass all cache checks

Cron (daily at 07:00)
-----
    0 7 * * * cd /path/to/wc2026_predictor && python3 auto_update.py >> data/outputs/auto_update.log 2>&1
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.utils import get_logger, save_df, load_df
from src.data_collection import fetch_international_results, fetch_elo_ratings

log = get_logger("auto_update")

# ── Tournament window ──────────────────────────────────────────────────────────
WC_START = date(2026, 6, 11)
WC_END   = date(2026, 7, 19)

# ── ESPN public API ────────────────────────────────────────────────────────────
_ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# All known name variants (ESPN, GitHub, and other sources) → config canonical.
# Covers both ESPN display names AND the spellings the martj42 GitHub repo uses,
# so that deduplication works even when sources disagree on team names.
_NAME_VARIANTS: dict[str, str] = {
    # ESPN-specific
    "USA":                          "United States",
    "Korea Republic":               "South Korea",
    "Republic of Korea":            "South Korea",
    "IR Iran":                      "Iran",
    "Côte d'Ivoire":                "Ivory Coast",
    "Cote d'Ivoire":                "Ivory Coast",
    "Bosnia-Herzegovina":           "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":         "Bosnia and Herzegovina",
    "Congo DR":                     "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cape Verde Islands":           "Cabo Verde",
    "Curacao":                      "Curaçao",
    "Czechia":                      "Czech Republic",
    # martj42 GitHub repo spellings that differ from config canonical
    "Turkey":                       "Türkiye",
    "Cape Verde":                   "Cabo Verde",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _fold(name: str) -> str:
    """Lowercase + strip diacritics for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_team(name: str) -> str:
    """Resolve any team name variant to config.ALL_TEAMS canonical spelling.

    Handles ESPN display names, martj42 GitHub spellings, and diacritic variants.
    Used on BOTH sides of dedup comparisons so that "Turkey" == "Türkiye".
    """
    if name in _NAME_VARIANTS:
        return _NAME_VARIANTS[name]
    folded = _fold(name)
    for team in config.ALL_TEAMS:
        if _fold(team) == folded:
            return team
    return name


# ── ESPN scraper ───────────────────────────────────────────────────────────────

def _fetch_espn_day(day: date) -> list[dict]:
    """Return completed WC 2026 match records for one calendar day from ESPN."""
    try:
        time.sleep(random.uniform(0.8, 1.5))
        resp = requests.get(
            _ESPN_SCOREBOARD,
            params={"dates": day.strftime("%Y%m%d")},
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("ESPN request failed for %s: %s", day, exc)
        return []

    rows = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_name = _normalize_team(home.get("team", {}).get("displayName", ""))
        away_name = _normalize_team(away.get("team", {}).get("displayName", ""))
        if not home_name or not away_name:
            continue

        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue

        rows.append({
            "date":       event.get("date", "")[:10] or day.isoformat(),
            "home_team":  home_name,
            "away_team":  away_name,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "FIFA World Cup",
            "city":       "North America",
            "country":    "USA/Canada/Mexico",
            "neutral":    True,
            "competitive": True,
            "_espn_status": status,
        })
    return rows


def _scrape_espn_range(since: date, until: date) -> pd.DataFrame:
    """Scrape ESPN for all completed WC 2026 matches between two dates (inclusive)."""
    log.info("  Scraping ESPN: %s → %s", since, until)
    rows: list[dict] = []
    current = since
    while current <= until:
        day_rows = _fetch_espn_day(current)
        if day_rows:
            log.info("    %s: %d completed match(es)", current, len(day_rows))
        rows.extend(day_rows)
        current += timedelta(days=1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    log.info("  ESPN total: %d completed WC match(es) found", len(df))
    return df


# ── results.csv ────────────────────────────────────────────────────────────────

def _dedup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate match rows caused by name variants or UTC date offsets.

    Two rows are treated as the same match when they share the same normalised
    team pair and their dates are within 1 day of each other AND their non-null
    scores agree (or one row has NaN scores).

    Strategy:
      1. Normalise team names in helper columns.
      2. Sort so rows WITH scores come before NaN rows (prefer real data).
      3. Drop exact (date, normalised-pair) duplicates keeping the first.
      4. For the remaining ±1-day duplicates, pick the row with scores.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["_h"] = df["home_team"].apply(_normalize_team)
    df["_a"] = df["away_team"].apply(_normalize_team)
    df["_pair"] = df.apply(lambda r: frozenset({r["_h"], r["_a"]}), axis=1)
    df["_has_score"] = df["home_score"].notna().astype(int)

    # Sort: scored rows first, then by date ascending
    df = df.sort_values(["_has_score", "date"], ascending=[False, True])

    # Drop exact-date duplicates (keeps scored row due to sort order)
    df = df.drop_duplicates(subset=["date", "_pair"], keep="first")

    # Drop ±1-day duplicates: for each remaining row, check if an earlier row
    # in the df covers the same match on an adjacent date.
    dates = pd.to_datetime(df["date"]).dt.date.tolist()
    pairs = df["_pair"].tolist()
    keep = [True] * len(df)
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df)):
            if not keep[j]:
                continue
            if pairs[i] == pairs[j] and abs((dates[i] - dates[j]).days) <= 1:
                keep[j] = False  # row i (with score) wins; drop row j

    df = df[keep].drop(columns=["_h", "_a", "_pair", "_has_score"])
    return df.reset_index(drop=True)


def update_results(force: bool = False) -> int:
    """Refresh results.csv from GitHub then patch in ESPN WC 2026 scores.

    Returns:
        Number of new rows appended to results.csv.
    """
    log.info("── results.csv ──")

    # Step 1: pull from GitHub (respects 24h TTL unless force=True)
    existing = fetch_international_results(force_refresh=force)
    before = len(existing)
    existing = _dedup_results(existing)
    if len(existing) < before:
        log.info("  Removed %d pre-existing duplicate(s) from results.csv", before - len(existing))
        save_df(existing, config.RAW_DIR / "results.csv")
    log.info("  GitHub rows: %d", len(existing))

    today = date.today()
    if today < WC_START:
        log.info("  Tournament hasn't started yet — nothing extra to scrape")
        return 0

    # Step 2: build a dedup index keyed by (date, frozenset of NORMALISED team names).
    # We also expand each date by ±1 day to absorb UTC vs local-time mismatches
    # (e.g. ESPN may record a 8 pm CDT match as the next UTC date).
    known: set[tuple[str, frozenset]] = set()
    for _, r in existing.iterrows():
        base = pd.to_datetime(str(r["date"])[:10]).date()
        h = _normalize_team(str(r["home_team"]))
        a = _normalize_team(str(r["away_team"]))
        pair = frozenset({h, a})
        for delta in (-1, 0, 1):
            known.add(((base + timedelta(days=delta)).isoformat(), pair))

    # Step 3: scrape ESPN for the WC window
    wc_df = _scrape_espn_range(WC_START, min(today, WC_END))
    if wc_df.empty:
        log.info("  No ESPN scores returned")
        return 0

    # Step 4: keep only rows not already in results.csv (after normalisation)
    novel_rows = [
        r for _, r in wc_df.iterrows()
        if (str(r["date"])[:10],
            frozenset({_normalize_team(str(r["home_team"])),
                       _normalize_team(str(r["away_team"]))})) not in known
    ]
    if not novel_rows:
        log.info("  All ESPN scores already present in results.csv")
        return 0

    novel_df = pd.DataFrame(novel_rows).drop(columns=["_espn_status"], errors="ignore")
    combined = pd.concat([existing, novel_df], ignore_index=True)
    save_df(combined, config.RAW_DIR / "results.csv")
    log.info("  +%d new row(s) → results.csv now has %d rows", len(novel_rows), len(combined))
    return len(novel_rows)


# ── wc2026_fixtures.csv ────────────────────────────────────────────────────────

def update_fixtures(force: bool = False) -> int:
    """Patch wc2026_fixtures.csv with actual scores for completed matches.

    Adds three columns (created if absent):
        score_t1  — goals scored by team1 (NaN if unplayed)
        score_t2  — goals scored by team2 (NaN if unplayed)
        status    — ESPN status string, e.g. STATUS_FINAL / scheduled

    Returns:
        Number of fixture rows updated with actual scores.
    """
    log.info("── wc2026_fixtures.csv ──")

    fixtures_path = config.RAW_DIR / "wc2026_fixtures.csv"
    if not fixtures_path.exists():
        from src.data_collection import fetch_wc2026_fixtures
        fixtures = fetch_wc2026_fixtures(force_refresh=True)
    else:
        fixtures = load_df(fixtures_path)

    # Ensure score/status columns exist
    for col, default in [("score_t1", float("nan")),
                          ("score_t2", float("nan")),
                          ("status", "scheduled")]:
        if col not in fixtures.columns:
            fixtures[col] = default

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.date
    today = date.today()

    if today < WC_START:
        log.info("  Tournament hasn't started yet — no fixtures to update")
        fixtures["date"] = fixtures["date"].astype(str)
        return 0

    past_mask    = fixtures["date"].apply(lambda d: d is not None and d <= today)
    unfilled     = fixtures["score_t1"].isna()
    rows_to_fill = fixtures[past_mask & unfilled] if not force else fixtures[past_mask]

    if rows_to_fill.empty:
        log.info("  All past fixtures already have scores")
        fixtures["date"] = fixtures["date"].astype(str)
        return 0

    log.info("  %d past fixture(s) need score data", len(rows_to_fill))

    min_date = rows_to_fill["date"].dropna().min()
    wc_df = _scrape_espn_range(min_date, min(today, WC_END))

    updated = 0
    if not wc_df.empty:
        # Index by (date, frozenset of team names) — safe for neutral venues
        # where ESPN's home/away assignment may differ from our fixture order.
        espn_index: dict[tuple, dict] = {}
        for _, r in wc_df.iterrows():
            d  = str(r["date"])[:10]
            h  = str(r["home_team"])
            a  = str(r["away_team"])
            key = (d, frozenset({h, a}))
            espn_index[key] = {
                h: int(r["home_score"]),
                a: int(r["away_score"]),
                "status": r.get("_espn_status", "STATUS_FINAL"),
            }

        for idx, row in rows_to_fill.iterrows():
            base_d = pd.to_datetime(str(row["date"])[:10]).date()
            t1 = str(row["team1"])
            t2 = str(row["team2"])
            pair = frozenset({t1, t2})
            # ±1 day tolerance: late-night kickoffs cross midnight UTC so ESPN
            # may record a match one calendar day ahead of the local fixture date.
            match = None
            for delta in (0, 1, -1):
                d_try = (base_d + timedelta(days=delta)).isoformat()
                match = espn_index.get((d_try, pair))
                if match:
                    break
            if match:
                fixtures.at[idx, "score_t1"] = match.get(t1, 0)
                fixtures.at[idx, "score_t2"] = match.get(t2, 0)
                fixtures.at[idx, "status"]   = match["status"]
                updated += 1

    fixtures["date"] = fixtures["date"].astype(str)
    fixtures = fixtures.sort_values("match_id").reset_index(drop=True)
    save_df(fixtures, fixtures_path)
    log.info("  Updated %d fixture(s) with actual scores", updated)
    return updated


# ── Cache invalidation ─────────────────────────────────────────────────────────

def invalidate_derived_caches() -> None:
    """Delete .meta sidecar files for datasets that depend on results.csv.

    h2h_stats and training_matrix will be rebuilt from scratch on the next
    pipeline call (build_match_features / retrain_and_simulate).
    """
    targets = [
        config.PROCESSED_DIR / "h2h_stats.csv.meta",
        config.PROCESSED_DIR / "training_matrix.csv.meta",
    ]
    for meta in targets:
        if meta.exists():
            meta.unlink()
            log.info("  Cache invalidated: %s", meta.name)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_once(force: bool = False) -> None:
    """Execute one full update cycle."""
    t0 = time.time()
    log.info("┌── Auto-update %s (force=%s)", date.today().isoformat(), force)

    new_results  = update_results(force=force)
    new_fixtures = update_fixtures(force=force)

    # Only bust H2H / training-matrix caches when we actually added new match data
    if new_results > 0:
        log.info("── Cache invalidation ──")
        invalidate_derived_caches()

    # Always refresh Elo ratings (lightweight scrape of eloratings.net)
    log.info("── elo_ratings.csv ──")
    try:
        fetch_elo_ratings(force_refresh=force)
        log.info("  Elo ratings refreshed")
    except Exception as exc:
        log.warning("  Elo refresh failed: %s", exc)

    log.info(
        "└── Done in %.1fs  (+%d results row(s), %d fixture(s) updated)",
        time.time() - t0, new_results, new_fixtures,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-update results.csv and wc2026_fixtures.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Run continuously (defaults to every 24 hours; see --interval)",
    )
    parser.add_argument(
        "--interval", type=float, default=24.0, metavar="HOURS",
        help="Hours between updates in --watch mode (default: 24)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass all cache checks — fully re-download everything",
    )
    args = parser.parse_args()

    if args.watch:
        log.info(
            "Watch mode active — updating every %.1f hour(s). Ctrl-C to stop.",
            args.interval,
        )
        while True:
            run_once(force=args.force)
            log.info("Next update in %.1f hour(s)…", args.interval)
            time.sleep(args.interval * 3600)
    else:
        run_once(force=args.force)


if __name__ == "__main__":
    main()
