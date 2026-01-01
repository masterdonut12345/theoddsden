# settlement.py
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests

from models import db, Parlay, ParlayLeg, WalletTxn

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

def utcnow():
    return datetime.now(timezone.utc)

def american_to_decimal(a: float) -> float:
    a = float(a)
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))

def _fetch_scores_for_events(sport_key: str, event_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Odds API v4 scores endpoint supports eventIds + daysFrom (1..3). :contentReference[oaicite:2]{index=2}
    """
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY missing")

    url = f"{ODDS_API_BASE}/sports/{sport_key}/scores/"
    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": 3,
        "dateFormat": "iso",
        "eventIds": ",".join(event_ids),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _extract_final_score(game_obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Tries to read home/away scores from Odds API scores payload.
    Different sports sometimes vary slightly; keep this defensive.
    """
    completed = bool(game_obj.get("completed"))
    scores = game_obj.get("scores") or []
    if not completed or not scores:
        return None

    # expected: [{"name": "...home team...", "score": "102"}, {"name":"...away...", "score":"98"}]
    try:
        # The object also has home_team/away_team names
        home_name = game_obj.get("home_team")
        away_name = game_obj.get("away_team")

        home_score = None
        away_score = None
        for s in scores:
            if s.get("name") == home_name:
                home_score = float(s.get("score"))
            elif s.get("name") == away_name:
                away_score = float(s.get("score"))

        if home_score is None or away_score is None:
            # fallback: take first two
            if len(scores) >= 2:
                home_score = float(scores[0].get("score"))
                away_score = float(scores[1].get("score"))
        if home_score is None or away_score is None:
            return None
        return home_score, away_score
    except Exception:
        return None

def _settle_leg(leg: ParlayLeg, home_score: float, away_score: float) -> str:
    """
    Returns WIN/LOSS/PUSH.
    Convention:
      - selection HOME_* refers to home team side
      - selection AWAY_* refers to away team side
      - spreads: line is the displayed team line (e.g. Celtics -3.5 for HOME_SPREAD means home line = -3.5)
      - totals: line is total_points
    """
    sel = leg.selection
    market = leg.market
    line = leg.line

    if market == "Moneyline":
        if home_score == away_score:
            return "PUSH"
        home_won = home_score > away_score
        if sel == "HOME_ML":
            return "WIN" if home_won else "LOSS"
        if sel == "AWAY_ML":
            return "WIN" if (not home_won) else "LOSS"
        return "PUSH"

    if market == "Spread":
        if line is None:
            return "PUSH"
        margin = home_score - away_score
        if sel == "HOME_SPREAD":
            # home covers if margin + line > 0 (line is negative for favorite)
            v = margin + float(line)
        else:
            # away spread line is for away team; easiest: away covers if (-margin) + away_line > 0
            v = (-margin) + float(line)
        if abs(v) < 1e-9:
            return "PUSH"
        return "WIN" if v > 0 else "LOSS"

    if market == "Total":
        if line is None:
            return "PUSH"
        total = home_score + away_score
        diff = total - float(line)
        if abs(diff) < 1e-9:
            return "PUSH"
        if sel == "OVER":
            return "WIN" if diff > 0 else "LOSS"
        if sel == "UNDER":
            return "WIN" if diff < 0 else "LOSS"
        return "PUSH"

    return "PUSH"

def run_settlement_once() -> Dict[str, Any]:
    """
    - Find pending parlays
    - For each sport_key, fetch scores for involved event_ids
    - Settle legs and parlays
    - Update wallet
    """
    pending = Parlay.query.filter_by(status="PENDING").all()
    if not pending:
        return {"ok": True, "pending": 0, "settled": 0}

    # collect event ids by sport
    by_sport: Dict[str, set] = {}
    legs_by_event: Dict[str, List[ParlayLeg]] = {}
    for p in pending:
        for leg in p.legs:
            by_sport.setdefault(leg.sport_key, set()).add(leg.event_id)
            legs_by_event.setdefault(leg.event_id, []).append(leg)

    # fetch scores per sport_key
    event_to_score: Dict[str, Tuple[float, float]] = {}
    for sport_key, ids in by_sport.items():
        ids_list = sorted(ids)
        if not ids_list:
            continue
        games = _fetch_scores_for_events(sport_key, ids_list)
        for g in games:
            eid = g.get("id")
            if not eid:
                continue
            fs = _extract_final_score(g)
            if fs is not None:
                event_to_score[str(eid)] = fs

    settled_parlays = 0

    # settle legs we can, then decide parlay
    for p in pending:
        all_done = True
        any_loss = False
        any_win = False
        any_push = False

        for leg in p.legs:
            if leg.result is not None:
                # already settled
                if leg.result == "LOSS": any_loss = True
                elif leg.result == "WIN": any_win = True
                elif leg.result == "PUSH": any_push = True
                continue

            fs = event_to_score.get(str(leg.event_id))
            if fs is None:
                all_done = False
                continue

            home_score, away_score = fs
            leg.result = _settle_leg(leg, home_score, away_score)

            if leg.result == "LOSS": any_loss = True
            elif leg.result == "WIN": any_win = True
            elif leg.result == "PUSH": any_push = True

        if not all_done:
            continue

        # finalize parlay
        p.settled_at = utcnow()

        if any_loss:
            p.status = "LOST"
            # no refund; stake already debited
        else:
            # all WIN or PUSH
            if not any_win and any_push:
                p.status = "PUSH"
                # refund stake
                p.user.balance += p.stake
                db.session.add(WalletTxn(user_id=p.user_id, kind="REFUND_PUSH", amount=+p.stake, note=f"Parlay {p.id} push refund"))
            else:
                p.status = "WON"
                # compute actual payout, voiding PUSH legs by treating them as decimal 1.0
                dec = 1.0
                for leg in p.legs:
                    if leg.result == "WIN":
                        dec *= american_to_decimal(leg.american)
                    elif leg.result == "PUSH":
                        dec *= 1.0
                payout = p.stake * dec
                p.potential_payout = payout
                p.user.balance += payout
                db.session.add(WalletTxn(user_id=p.user_id, kind="CREDIT_WIN", amount=+payout, note=f"Parlay {p.id} won payout"))

        settled_parlays += 1

    db.session.commit()
    return {"ok": True, "pending": len(pending), "settled": settled_parlays}
