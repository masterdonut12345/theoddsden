#!/usr/bin/env python3
import os
import json
import math
import re
import difflib
import threading
import time
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from sqlalchemy.exc import IntegrityError
from sqlalchemy import desc, func, case, text

from flask import (
    Flask, render_template, abort, request, jsonify,
    redirect, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# Optional joblib (don't crash app if missing)
try:
    import joblib
except Exception:
    joblib = None


# ----------------------------
# App + Config
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

LOCAL_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "America/Chicago"))

DATA_GAMES = os.environ.get("DATA_CSV", "nba_nfl_games_with_odds_and_team_metrics.csv")
TRENDS_DIR = os.environ.get("TRENDS_DIR", ".")
TRENDS_PATTERN = os.environ.get("TRENDS_PATTERN", "team_trends_last{days}d.csv")
TRENDS_FALLBACK = os.environ.get("TRENDS_CSV", "team_trends_last30d.csv")

DEFAULT_SPAN_DAYS = int(os.environ.get("DEFAULT_SPAN_DAYS", "30"))
SPAN_CHOICES = [7, 30, 180, 365, 730]

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///site.db")
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

DEBUG_ML = os.environ.get("DEBUG_ML", "0") == "1"
logging.basicConfig(level=logging.INFO if DEBUG_ML else logging.WARNING)
log = logging.getLogger("oddsinsight")


# ----------------------------
# Auth / Login
# ----------------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"


# ----------------------------
# DB Safety: rollback on request errors
# ----------------------------
@app.teardown_request
def _teardown_request(exc):
    # Prevent "InFailedSqlTransaction" from poisoning future requests
    if exc is not None:
        try:
            db.session.rollback()
        except Exception:
            pass


# ----------------------------
# Models
# ----------------------------
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    balance = db.Column(db.Integer, nullable=False, default=1000)

    # For daily claim button:
    last_claimed_at = db.Column(db.DateTime, nullable=True)

    def set_password(self, pw: str) -> None:
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self.password_hash, pw)


class Parlay(db.Model):
    __tablename__ = "parlays"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user = db.relationship("User", backref="parlays")

    stake = db.Column(db.Integer, nullable=False)
    decimal_odds = db.Column(db.Float, nullable=True)
    american_odds = db.Column(db.Integer, nullable=True)

    status = db.Column(db.String(16), nullable=False, default="PENDING")  # PENDING/WON/LOST/PUSH/VOID
    payout = db.Column(db.Integer, nullable=False, default=0)

    placed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    settled_at = db.Column(db.DateTime, nullable=True)

    legs = db.relationship("ParlayLeg", cascade="all, delete-orphan", backref="parlay")


class ParlayLeg(db.Model):
    __tablename__ = "parlay_legs"
    id = db.Column(db.Integer, primary_key=True)

    parlay_id = db.Column(db.Integer, db.ForeignKey("parlays.id"), nullable=False, index=True)

    event_id = db.Column(db.String(64), nullable=False, index=True)
    league = db.Column(db.String(8), nullable=False)

    market = db.Column(db.String(16), nullable=False)
    selection = db.Column(db.String(32), nullable=False)

    label = db.Column(db.String(200), nullable=False)
    matchup = db.Column(db.String(200), nullable=False)

    american = db.Column(db.Integer, nullable=False)
    point = db.Column(db.Float, nullable=True)


class EventResult(db.Model):
    __tablename__ = "event_results"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    league = db.Column(db.String(8), nullable=True)

    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)

    winner = db.Column(db.String(8), nullable=True)  # "HOME" / "AWAY"
    final = db.Column(db.Boolean, nullable=False, default=False)

    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


# ----------------------------
# Admin token helpers
# ----------------------------
def _require_admin_token() -> bool:
    token = os.environ.get("ADMIN_TOKEN", "")
    supplied = (request.args.get("token") or "").strip()
    return bool(token) and supplied == token

def _admin_unauth():
    return jsonify({"ok": False, "error": "unauthorized"}), 401


# ----------------------------
# DB migrate helpers (adds missing columns safely)
# ----------------------------
def ensure_schema():
    """
    create_all + best-effort add missing columns for SQLite/Postgres.
    This prevents: "no such column users.last_claimed_at".
    """
    db.create_all()

    # users.last_claimed_at
    try:
        # Postgres: IF NOT EXISTS ok
        db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_claimed_at TIMESTAMP NULL"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        # SQLite: no IF NOT EXISTS (often). Try plain ALTER; ignore if already exists.
        try:
            db.session.execute(text("ALTER TABLE users ADD COLUMN last_claimed_at TIMESTAMP NULL"))
            db.session.commit()
        except Exception:
            db.session.rollback()


# ----------------------------
# Admin endpoints
# ----------------------------
@app.get("/admin/init_db")
def admin_init_db():
    if not _require_admin_token():
        return _admin_unauth()
    try:
        with app.app_context():
            db.create_all()
        return jsonify({"ok": True, "message": "db.create_all() completed"})
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({"ok": False, "error": "init_failed", "detail": str(e)}), 500


@app.get("/admin/migrate_db")
def admin_migrate_db():
    if not _require_admin_token():
        return _admin_unauth()
    try:
        with app.app_context():
            ensure_schema()
        return jsonify({"ok": True, "message": "migrate completed (create_all + add missing columns)"})
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({"ok": False, "error": "migrate_failed", "detail": str(e)}), 500


@app.get("/admin/delete_user")
def admin_delete_user():
    if not _require_admin_token():
        return _admin_unauth()

    username = (request.args.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "missing_username"}), 400

    try:
        u = User.query.filter_by(username=username).first()
        if not u:
            return jsonify({"ok": False, "error": "user_not_found"}), 404

        parlay_ids = [p.id for p in Parlay.query.filter_by(user_id=u.id).all()]
        if parlay_ids:
            ParlayLeg.query.filter(ParlayLeg.parlay_id.in_(parlay_ids)).delete(synchronize_session=False)
            Parlay.query.filter(Parlay.id.in_(parlay_ids)).delete(synchronize_session=False)

        db.session.delete(u)
        db.session.commit()
        return jsonify({"ok": True, "deleted": username})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": "delete_failed", "detail": str(e)}), 500


@app.get("/admin/grant_coins")
def admin_grant_coins():
    if not _require_admin_token():
        return _admin_unauth()

    username = (request.args.get("username") or "").strip()
    amount_raw = (request.args.get("amount") or "").strip()

    if not username:
        return jsonify({"ok": False, "error": "missing_username"}), 400
    try:
        amount = int(amount_raw)
    except Exception:
        return jsonify({"ok": False, "error": "invalid_amount"}), 400

    try:
        u = User.query.filter_by(username=username).first()
        if not u:
            return jsonify({"ok": False, "error": "user_not_found"}), 404

        try:
            u = db.session.query(User).filter(User.id == u.id).with_for_update().one()
        except Exception:
            u = db.session.query(User).filter(User.id == u.id).one()

        before = int(u.balance or 0)
        after = before + amount
        if after < 0:
            after = 0
        u.balance = after
        db.session.commit()

        return jsonify({"ok": True, "username": u.username, "before": before, "amount": amount, "after": after})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": "grant_failed", "detail": str(e)}), 500


@app.post("/admin/upsert_result")
def admin_upsert_result():
    if not _require_admin_token():
        return _admin_unauth()

    data = request.get_json(silent=True) or {}
    event_id = str(data.get("event_id") or "").strip()
    if not event_id:
        return jsonify({"ok": False, "error": "missing_event_id"}), 400

    try:
        r = EventResult.query.filter_by(event_id=event_id).first()
        if r is None:
            r = EventResult(event_id=event_id)
            db.session.add(r)

        r.league = (data.get("league") or r.league)
        r.home_score = data.get("home_score", r.home_score)
        r.away_score = data.get("away_score", r.away_score)
        r.winner = data.get("winner", r.winner)
        r.final = bool(data.get("final", r.final))
        r.updated_at = datetime.utcnow()

        db.session.commit()
        return jsonify({"ok": True, "event_id": r.event_id, "final": r.final})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": "upsert_failed", "detail": str(e)}), 500


# ----------------------------
# Parlay settlement logic
# ----------------------------
def _leg_won(leg: ParlayLeg, r: EventResult):
    if not r or not r.final:
        return None
    if r.home_score is None or r.away_score is None or r.winner is None:
        return None

    market = (leg.market or "").strip()
    sel = (leg.selection or "").strip()

    home_score = int(r.home_score)
    away_score = int(r.away_score)
    diff_home = home_score - away_score
    total_pts = home_score + away_score

    if market == "Moneyline":
        if sel == "HOME_ML":
            return r.winner == "HOME"
        if sel == "AWAY_ML":
            return r.winner == "AWAY"
        return None

    if market == "Spread":
        if leg.point is None:
            return None
        if sel == "HOME_SPREAD":
            return (diff_home + float(leg.point)) > 0.0
        if sel == "AWAY_SPREAD":
            return ((-diff_home) + float(leg.point)) > 0.0
        return None

    if market == "Total":
        if leg.point is None:
            return None
        if sel == "OVER":
            return total_pts > float(leg.point)
        if sel == "UNDER":
            return total_pts < float(leg.point)
        return None

    return None


def _parlay_payout(stake: int, decimal_odds: float) -> int:
    try:
        return int(round(float(stake) * float(decimal_odds)))
    except Exception:
        return 0


def settle_parlays_once(limit: int = 100) -> int:
    """
    Settles up to `limit` pending parlays whose legs all have final EventResult rows.
    Uses a Postgres advisory lock to prevent double settlement across workers.
    """
    got_lock = False
    try:
        row = db.session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": 987654321}).fetchone()
        got_lock = bool(row and row[0])
        if not got_lock:
            return 0
    except Exception:
        got_lock = False

    settled = 0
    try:
        pending = (
            db.session.query(Parlay)
            .filter(Parlay.status == "PENDING")
            .order_by(Parlay.placed_at.asc())
            .limit(limit)
            .all()
        )

        for p in pending:
            legs = (
                db.session.query(ParlayLeg)
                .filter(ParlayLeg.parlay_id == p.id)
                .all()
            )
            if not legs:
                continue

            event_ids = list({str(l.event_id) for l in legs})
            results = (
                db.session.query(EventResult)
                .filter(EventResult.event_id.in_(event_ids))
                .all()
            )
            res_by_event = {r.event_id: r for r in results}

            outcomes = []
            ready = True
            for l in legs:
                r = res_by_event.get(str(l.event_id))
                won = _leg_won(l, r)
                if won is None:
                    ready = False
                    break
                outcomes.append(bool(won))

            if not ready:
                continue

            p.settled_at = datetime.utcnow()

            if all(outcomes):
                p.status = "WON"
                p.payout = _parlay_payout(p.stake, p.decimal_odds or 0.0)
                u = db.session.query(User).filter(User.id == p.user_id).one()
                u.balance += int(p.payout)
            else:
                p.status = "LOST"
                p.payout = 0

            settled += 1

        db.session.flush()
        return settled

    finally:
        if got_lock:
            try:
                db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": 987654321})
            except Exception:
                pass


# ----------------------------
# Public scoreboard API (no key required) -> EventResult fetcher
# ----------------------------
ESPN_SCOREBOARD_BASE = "https://site.api.espn.com/apis/v2/sports"

LEAGUE_TO_ESPN_PATH = {
    "NBA": ("basketball", "nba"),
    "NFL": ("football", "nfl"),
}

def _parse_utc_datetime(val):
    try:
        if val is None:
            return None
        if isinstance(val, datetime):
            dt = val
        else:
            s = str(val)
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

def _event_lookup_for_league(league: str, event_ids: list[str]):
    games = load_data()
    if games is None or getattr(games, "empty", True):
        return {}

    df = games[games.get("league") == league] if hasattr(games, "get") else games
    try:
        df = games[games["league"] == league]
    except Exception:
        pass

    if event_ids:
        try:
            df = df[df["event_id"].isin(event_ids)]
        except Exception:
            pass

    lookup = {}
    for _, row in (df.iterrows() if hasattr(df, "iterrows") else []):
        try:
            eid = str(row.get("event_id"))
        except Exception:
            eid = str(row["event_id"])
        if not eid:
            continue
        home_team = str(row.get("home_team", "") or "").strip()
        away_team = str(row.get("away_team", "") or "").strip()
        commence = _parse_utc_datetime(row.get("commence_time_utc")) if hasattr(row, "get") else None
        lookup[eid] = {
            "home_team": home_team,
            "away_team": away_team,
            "norm_home": _norm_team(home_team),
            "norm_away": _norm_team(away_team),
            "commence": commence,
        }
    return lookup

def fetch_scores_from_public_api(league: str, event_ids: list[str], days_from: int = 3) -> list[dict]:
    path = LEAGUE_TO_ESPN_PATH.get(league)
    if not path:
        return []

    event_lookup = _event_lookup_for_league(league, event_ids)
    if not event_lookup:
        return []

    name_index: dict[tuple[str, str], list[str]] = {}
    for eid, info in event_lookup.items():
        name_index.setdefault((info["norm_home"], info["norm_away"]), []).append(eid)

    sport_part, league_part = path
    today = datetime.now(timezone.utc).date()
    dates = [today - timedelta(days=i) for i in range(max(1, days_from))]

    out: list[dict] = []
    seen: set[str] = set()

    for d in dates:
        url = f"{ESPN_SCOREBOARD_BASE}/{sport_part}/{league_part}/scoreboard"
        try:
            r = requests.get(url, params={"dates": d.strftime("%Y%m%d")}, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception:
            continue

        events = payload.get("events") or []
        for ev in events:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            def _team_name(obj):
                team = obj.get("team") or {}
                return (
                    team.get("displayName")
                    or team.get("shortDisplayName")
                    or team.get("name")
                    or ""
                )

            home_name = _team_name(home)
            away_name = _team_name(away)
            key = (_norm_team(home_name), _norm_team(away_name))
            candidate_ids = name_index.get(key, [])
            if not candidate_ids:
                continue

            start_dt = _parse_utc_datetime(comp.get("date") or ev.get("date"))
            chosen_id = None
            best_delta = None
            for cid in candidate_ids:
                info = event_lookup.get(cid) or {}
                commence = info.get("commence")
                delta = None
                if commence and start_dt:
                    delta = abs((start_dt - commence).total_seconds())
                if chosen_id is None or (delta is not None and (best_delta is None or delta < best_delta)):
                    chosen_id = cid
                    best_delta = delta

            if not chosen_id or chosen_id in seen:
                continue

            info = event_lookup.get(chosen_id) or {}
            status_type = (ev.get("status") or {}).get("type") or {}
            completed = bool(status_type.get("completed"))

            out.append({
                "id": chosen_id,
                "completed": completed,
                "home_team": info.get("home_team", home_name),
                "away_team": info.get("away_team", away_name),
                "scores": [
                    {"name": info.get("home_team", home_name), "score": home.get("score")},
                    {"name": info.get("away_team", away_name), "score": away.get("score")},
                ],
            })
            seen.add(chosen_id)

    return out

def upsert_event_result_from_score_item(item: dict, league: str) -> bool:
    """
    Returns True if we upserted a FINAL result, else False.
    """
    try:
        event_id = str(item.get("id") or "").strip()
        if not event_id:
            return False

        completed = bool(item.get("completed", False))
        scores_list = item.get("scores") or []
        home_team = str(item.get("home_team") or "")
        away_team = str(item.get("away_team") or "")

        # scores_list usually: [{"name":"Team","score":"110"}, ...]
        score_by_name = {}
        for s in scores_list:
            nm = str(s.get("name") or "")
            sc = s.get("score")
            try:
                sc = int(sc)
            except Exception:
                sc = None
            if nm:
                score_by_name[nm] = sc

        home_score = score_by_name.get(home_team)
        away_score = score_by_name.get(away_team)

        winner = None
        if completed and home_score is not None and away_score is not None:
            if home_score > away_score:
                winner = "HOME"
            elif away_score > home_score:
                winner = "AWAY"

        r = EventResult.query.filter_by(event_id=event_id).first()
        if r is None:
            r = EventResult(event_id=event_id)
            db.session.add(r)

        r.league = league
        r.home_score = home_score
        r.away_score = away_score
        r.winner = winner
        r.final = completed
        r.updated_at = datetime.utcnow()

        return bool(completed)
    except Exception:
        return False

def fetch_and_upsert_results_for_pending_parlays(days_from: int = 3) -> dict:
    out = {"ok": True, "leagues": {}, "final_upserts": 0}

    pending_leg_rows = (
        db.session.query(ParlayLeg.league, ParlayLeg.event_id)
        .join(Parlay, Parlay.id == ParlayLeg.parlay_id)
        .filter(Parlay.status == "PENDING")
        .distinct()
        .all()
    )

    leagues = sorted({str(lg) for (lg, _) in pending_leg_rows if lg})
    if not leagues:
        return out

    for league in leagues:
        event_ids = [str(eid) for (lg, eid) in pending_leg_rows if str(lg) == league]
        try:
            items = fetch_scores_from_public_api(league, event_ids, days_from=days_from)
        except Exception as e:
            out["leagues"][league] = {"ok": False, "error": str(e), "fetched": 0, "final_upserts": 0}
            continue

        final_upserts = 0
        for it in items:
            if upsert_event_result_from_score_item(it, league=league):
                final_upserts += 1

        out["leagues"][league] = {"ok": True, "fetched": len(items), "final_upserts": final_upserts}
        out["final_upserts"] += final_upserts

    return out

def run_results_and_settle_once(fetch_days_from: int = 3, settle_limit: int = 200) -> dict:
    report = {"ok": True, "fetch": {}, "settled": 0}

    try:
        report["fetch"] = fetch_and_upsert_results_for_pending_parlays(days_from=fetch_days_from)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        report["ok"] = False
        report["fetch_error"] = str(e)
        return report

    try:
        n = settle_parlays_once(limit=settle_limit)
        db.session.commit()
        report["settled"] = int(n or 0)
    except Exception as e:
        db.session.rollback()
        report["ok"] = False
        report["settle_error"] = str(e)

    return report


@app.get("/admin/settle_parlays")
def admin_settle_parlays():
    if not _require_admin_token():
        return _admin_unauth()
    try:
        n = settle_parlays_once(limit=200)
        db.session.commit()
        return jsonify({"ok": True, "settled": n})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": "settle_failed", "detail": str(e)}), 500


@app.get("/admin/run_worker_tick")
def admin_run_worker_tick():
    """
    Manual HTTPS trigger: fetch results -> upsert -> settle
    """
    if not _require_admin_token():
        return _admin_unauth()
    try:
        fetch_days = int(request.args.get("daysFrom", "3"))
    except Exception:
        fetch_days = 3
    try:
        with app.app_context():
            rep = run_results_and_settle_once(fetch_days_from=fetch_days, settle_limit=200)
        return jsonify(rep)
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({"ok": False, "error": "tick_failed", "detail": str(e)}), 500


@app.get("/admin/debug_parlay")
def admin_debug_parlay():
    if not _require_admin_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    try:
        parlay_id = int(request.args.get("parlay_id", "0"))
    except Exception:
        return jsonify({"ok": False, "error": "invalid_parlay_id"}), 400

    p = db.session.query(Parlay).filter(Parlay.id == parlay_id).first()
    if not p:
        return jsonify({"ok": False, "error": "parlay_not_found"}), 404

    legs = db.session.query(ParlayLeg).filter(ParlayLeg.parlay_id == p.id).all()
    event_ids = sorted({str(l.event_id) for l in legs})
    results = db.session.query(EventResult).filter(EventResult.event_id.in_(event_ids)).all()
    res_by_event = {str(r.event_id): r for r in results}

    leg_debug = []
    ready = True
    for l in legs:
        r = res_by_event.get(str(l.event_id))
        won = _leg_won(l, r)  # True/False/None
        if won is None:
            ready = False
        leg_debug.append({
            "event_id": str(l.event_id),
            "market": l.market,
            "selection": l.selection,
            "point": l.point,
            "result_found": bool(r),
            "final": (bool(r.final) if r else None),
            "home_score": (r.home_score if r else None),
            "away_score": (r.away_score if r else None),
            "winner": (r.winner if r else None),
            "won": won,
        })

    return jsonify({
        "ok": True,
        "parlay": {
            "id": p.id,
            "status": p.status,
            "stake": p.stake,
            "decimal_odds": p.decimal_odds,
            "placed_at": p.placed_at.isoformat() if p.placed_at else None,
            "settled_at": p.settled_at.isoformat() if p.settled_at else None,
        },
        "ready_to_settle": ready,
        "legs": leg_debug,
        "missing_event_results": [eid for eid in event_ids if eid not in res_by_event],
    })


# ----------------------------
# Background worker: fetch results + settle
# ----------------------------
_worker_thread = None
_worker_stop = threading.Event()
_worker_lock = threading.Lock()

def _worker_loop(interval_seconds: int):
    fetch_days = int(os.environ.get("WORKER_FETCH_DAYSFROM", "3"))
    while not _worker_stop.is_set():
        try:
            with app.app_context():
                run_results_and_settle_once(fetch_days_from=fetch_days, settle_limit=200)
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass

        for _ in range(max(1, interval_seconds)):
            if _worker_stop.is_set():
                break
            time.sleep(1)

def start_worker_if_needed():
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        interval = int(os.environ.get("WORKER_INTERVAL_SECONDS", "60"))
        _worker_stop.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(interval,),
            daemon=True,
            name="results+settle-worker",
        )
        _worker_thread.start()

@app.before_request
def _ensure_worker_started():
    # Starts once when the first HTTP request hits this process
    if os.environ.get("RUN_WORKER_IN_WEB", "1") == "1":
        start_worker_if_needed()
    if os.environ.get("RUN_SCRAPER_IN_WEB", "1") == "1":
        start_scraper_if_needed()

import atexit
@atexit.register
def _stop_worker():
    _worker_stop.set()


# ----------------------------
# Background odds scraper worker (runs odd_scraper.py) to refresh odds
# ----------------------------
SCRAPE_ENABLED = os.environ.get("SCRAPE_ENABLED", "1") == "1"
ODDS_SCRAPER_PATH = os.environ.get("ODDS_SCRAPER_PATH", "odd_scraper.py")
SCRAPE_OUTDIR = os.environ.get("OUTDIR", ".")  # odds_scraper.py already reads OUTDIR; pass too
SCRAPE_GAMES_INTERVAL_SECONDS = int(os.environ.get("SCRAPE_GAMES_INTERVAL_SECONDS", str(60 * 120)))  # default 2 hours
SCRAPE_GAMES_MIN_GAP_SECONDS = int(os.environ.get("SCRAPE_GAMES_MIN_GAP_SECONDS", "300"))  # avoid rapid repeats
SCRAPE_SUBPROCESS_TIMEOUT = int(os.environ.get("SCRAPE_SUBPROCESS_TIMEOUT", "1800"))  # 30 min
SCRAPE_LOG_STDOUT = os.environ.get("SCRAPE_LOG_STDOUT", "0") == "1"
LOCK_KEY_SCRAPE_GAMES = int(os.environ.get("SCRAPE_LOCK_KEY_GAMES", "271828182"))

_scraper_thread = None
_scraper_stop = threading.Event()
_scraper_lock = threading.Lock()

_scrape_state = {
    "games": {"last_started_utc": None, "last_finished_utc": None, "last_ok": None, "last_error": None},
}

def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _pg_try_lock(key: int) -> bool:
    try:
        row = db.session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": int(key)}).fetchone()
        return bool(row and row[0])
    except Exception:
        return False

def _pg_unlock(key: int) -> None:
    try:
        db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": int(key)})
    except Exception:
        pass

def _run_odds_scraper_games() -> bool:
    """
    Runs odd_scraper.py in games mode to refresh the odds CSV.
    """
    if not os.path.exists(ODDS_SCRAPER_PATH):
        raise RuntimeError(f"ODDS_SCRAPER_PATH not found: {ODDS_SCRAPER_PATH}")

    env = dict(os.environ)
    env["OUTDIR"] = SCRAPE_OUTDIR  # align with odds_scraper.py defaults

    cmd = [
        "python3",
        ODDS_SCRAPER_PATH,
        "--mode", "games",
        "--outdir", SCRAPE_OUTDIR,
    ]

    started = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=SCRAPE_SUBPROCESS_TIMEOUT,
    )
    dur = time.time() - started

    if SCRAPE_LOG_STDOUT and (proc.stdout or proc.stderr):
        log.warning("[scraper:games] stdout:\n%s", (proc.stdout or "").strip())
        log.warning("[scraper:games] stderr:\n%s", (proc.stderr or "").strip())
    elif proc.returncode != 0:
        log.warning("[scraper:games] stderr:\n%s", (proc.stderr or "").strip())

    if proc.returncode != 0:
        raise RuntimeError(f"odd_scraper.py games failed (code={proc.returncode}, dur={dur:.1f}s)")

    return True

def _scrape_games_once() -> None:
    try:
        with app.app_context():
            got = _pg_try_lock(LOCK_KEY_SCRAPE_GAMES)
            if not got:
                return
            try:
                _scrape_state["games"]["last_started_utc"] = _utc_now_iso()
                _scrape_state["games"]["last_error"] = None
                ok = _run_odds_scraper_games()
                _scrape_state["games"]["last_ok"] = bool(ok)
                _scrape_state["games"]["last_finished_utc"] = _utc_now_iso()
                load_data(force_reload=True)
            except Exception as e:
                _scrape_state["games"]["last_ok"] = False
                _scrape_state["games"]["last_error"] = str(e)
                _scrape_state["games"]["last_finished_utc"] = _utc_now_iso()
            finally:
                _pg_unlock(LOCK_KEY_SCRAPE_GAMES)
    except Exception:
        # keep worker alive even if a cycle fails
        pass

def _scraper_loop():
    """
    Run once at startup, then every SCRAPE_GAMES_INTERVAL_SECONDS (default 2 hours).
    """
    last_run_ts = 0.0
    first_run = True

    while not _scraper_stop.is_set():
        now_ts = time.time()
        should_run = False

        if first_run:
            should_run = True
            first_run = False
        elif (now_ts - last_run_ts) >= max(60, SCRAPE_GAMES_INTERVAL_SECONDS, SCRAPE_GAMES_MIN_GAP_SECONDS):
            should_run = True

        if should_run:
            _scrape_games_once()
            last_run_ts = time.time()

        for _ in range(10):
            if _scraper_stop.is_set():
                break
            time.sleep(1)

def start_scraper_if_needed():
    global _scraper_thread
    if not SCRAPE_ENABLED:
        return
    if _scraper_thread is not None and _scraper_thread.is_alive():
        return
    with _scraper_lock:
        if _scraper_thread is not None and _scraper_thread.is_alive():
            return
        _scraper_stop.clear()
        _scraper_thread = threading.Thread(
            target=_scraper_loop,
            daemon=True,
            name="odds-scraper",
        )
        _scraper_thread.start()

@atexit.register
def _stop_scraper():
    _scraper_stop.set()


# ----------------------------
# Leaderboard
# ----------------------------
@app.get("/leaderboard")
def leaderboard_page():
    agg = (
        db.session.query(
            Parlay.user_id.label("uid"),
            func.count(Parlay.id).label("parlays_placed"),
            func.sum(case((Parlay.status == "WON", 1), else_=0)).label("wins"),
            func.sum(case((Parlay.status.in_(["WON", "LOST"]), 1), else_=0)).label("settled"),
        )
        .group_by(Parlay.user_id)
        .subquery()
    )

    rows = (
        db.session.query(
            User,
            func.coalesce(agg.c.parlays_placed, 0).label("parlays_placed"),
            func.coalesce(agg.c.wins, 0).label("wins"),
            func.coalesce(agg.c.settled, 0).label("settled"),
        )
        .outerjoin(agg, agg.c.uid == User.id)
        .order_by(User.balance.desc(), User.username.asc())
        .limit(100)
        .all()
    )

    top_users = []
    for (u, parlays_placed, wins, settled) in rows:
        hit_pct = None
        if settled and int(settled) > 0:
            hit_pct = (float(wins) / float(settled)) * 100.0
        top_users.append({
            "username": u.username,
            "balance": int(u.balance or 0),
            "parlays_placed": int(parlays_placed or 0),
            "hit_pct": hit_pct,
        })

    return render_template("leaderboard.html", top_users=top_users)


@app.get("/api/leaderboard")
def leaderboard_api():
    agg = (
        db.session.query(
            Parlay.user_id.label("uid"),
            func.count(Parlay.id).label("parlays_placed"),
            func.sum(case((Parlay.status == "WON", 1), else_=0)).label("wins"),
            func.sum(case((Parlay.status.in_(["WON", "LOST"]), 1), else_=0)).label("settled"),
        )
        .group_by(Parlay.user_id)
        .subquery()
    )

    rows = (
        db.session.query(
            User,
            func.coalesce(agg.c.parlays_placed, 0).label("parlays_placed"),
            func.coalesce(agg.c.wins, 0).label("wins"),
            func.coalesce(agg.c.settled, 0).label("settled"),
        )
        .outerjoin(agg, agg.c.uid == User.id)
        .order_by(User.balance.desc(), User.username.asc())
        .limit(100)
        .all()
    )

    out = []
    for i, (u, parlays_placed, wins, settled) in enumerate(rows):
        hit_pct = None
        if settled and int(settled) > 0:
            hit_pct = (float(wins) / float(settled)) * 100.0
        out.append({
            "rank": i + 1,
            "username": u.username,
            "balance": int(u.balance or 0),
            "parlays_placed": int(parlays_placed or 0),
            "hit_pct": hit_pct,
        })

    return jsonify({"ok": True, "count": len(out), "users": out})


# ----------------------------
# Daily coin claim (1x/day)
# ----------------------------
@app.post("/api/claim_daily")
@login_required
def api_claim_daily():
    daily_amount = int(os.environ.get("DAILY_CLAIM_COINS", "250"))
    now = datetime.now(timezone.utc)

    try:
        u_db = User.query.filter_by(id=current_user.id).first()
        if not u_db:
            return jsonify({"ok": False, "error": "user_not_found"}), 404

        try:
            u_db = db.session.query(User).filter(User.id == u_db.id).with_for_update().one()
        except Exception:
            pass

        last = u_db.last_claimed_at
        if last is not None:
            if last.replace(tzinfo=timezone.utc).date() == now.date():
                return jsonify({
                    "ok": False,
                    "error": "already_claimed",
                    "message": "You already claimed today.",
                    "balance": int(u_db.balance or 0),
                }), 400

        u_db.balance = int(u_db.balance or 0) + daily_amount
        u_db.last_claimed_at = now.replace(tzinfo=None)  # store naive UTC
        db.session.commit()

        return jsonify({"ok": True, "amount": daily_amount, "balance": int(u_db.balance or 0)})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": "claim_failed", "detail": str(e)}), 500


# ----------------------------
# Metrics config
# ----------------------------
METRICS = {
    "rolling_win_pct_10": {"label": "Rolling win %", "kind": "pct"},
    "point_diff": {"label": "Point differential", "kind": "num"},
    "points_for": {"label": "Points scored (per game)", "kind": "num"},
    "points_against": {"label": "Points allowed (per game)", "kind": "num"},
    "cum_wins": {"label": "Cumulative wins", "kind": "num"},
    "cum_losses": {"label": "Cumulative losses", "kind": "num"},
    "rolling_point_diff_10": {"label": "Rolling point diff", "kind": "num"},
    "rolling_pf_10": {"label": "Rolling points for", "kind": "num"},
    "rolling_pa_10": {"label": "Rolling points against", "kind": "num"},
}


# ----------------------------
# Load ML models (optional)
# ----------------------------
ML_MODEL_NBA = None
ML_MODEL_NFL = None
if joblib is not None:
    try:
        if os.path.exists("moneyline_model_NBA.joblib"):
            ML_MODEL_NBA = joblib.load("moneyline_model_NBA.joblib")
        if os.path.exists("moneyline_model_NFL.joblib"):
            ML_MODEL_NFL = joblib.load("moneyline_model_NFL.joblib")
    except Exception as e:
        log.warning("Failed loading joblib models: %s", e)
        ML_MODEL_NBA = None
        ML_MODEL_NFL = None


# ----------------------------
# Caches + helpers
# ----------------------------
_games_cache = None
_trends_cache_by_days = {}
_latest_team_features_cache = {}
_team_name_index = {}

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def _norm_team(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _resolve_team_name(league: str, team: str) -> str:
    league = str(league or "")
    team = str(team or "")
    if not league or not team:
        return team
    idx = _team_name_index.get(league) or {}
    if not idx:
        return team
    key = _norm_team(team)
    if key in idx:
        return idx[key]
    keys = list(idx.keys())
    if not keys:
        return team
    matches = difflib.get_close_matches(key, keys, n=1, cutoff=0.80)
    if matches:
        return idx[matches[0]]
    return team

def _trend_file_for_days(days: int) -> str:
    return os.path.join(TRENDS_DIR, TRENDS_PATTERN.format(days=days))

def _load_trends_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
    return df

def load_data(force_reload: bool = False):
    global _games_cache, _trends_cache_by_days, _latest_team_features_cache, _team_name_index

    if _games_cache is None or force_reload:
        if not os.path.exists(DATA_GAMES):
            _games_cache = pd.DataFrame()
        else:
            games = pd.read_csv(DATA_GAMES)
            games = games.where(pd.notnull(games), None)

            for col in ["commence_time_utc", "commence_time_local"]:
                if col in games.columns:
                    games[col] = pd.to_datetime(games[col], utc=True, errors="coerce")

            numeric_cols = [
                "home_ml_best", "away_ml_best",
                "home_ml_prob_devig_med", "away_ml_prob_devig_med",
                "spread_home_point", "spread_home_price_best",
                "spread_away_point", "spread_away_price_best",
                "total_points", "total_over_price_best", "total_under_price_best",
                "home_wins", "home_losses", "home_ties", "home_winPercent",
                "home_pointsFor", "home_pointsAgainst", "home_ppg", "home_opp_ppg", "home_avg_diff",
                "away_wins", "away_losses", "away_ties", "away_winPercent",
                "away_pointsFor", "away_pointsAgainst", "away_ppg", "away_opp_ppg", "away_avg_diff",
            ]
            for c in numeric_cols:
                if c in games.columns:
                    games[c] = pd.to_numeric(games[c], errors="coerce")

            games = games.where(pd.notnull(games), None)

            if "commence_time_utc" in games.columns:
                def _to_local_str(ts):
                    try:
                        if ts is None or pd.isna(ts):
                            return None
                        if not isinstance(ts, pd.Timestamp):
                            ts = pd.to_datetime(ts, utc=True, errors="coerce")
                        if ts is None or pd.isna(ts):
                            return None
                        dt = ts.to_pydatetime().astimezone(LOCAL_TZ)
                        return dt.strftime("%a %b %d • %I:%M %p %Z")
                    except Exception:
                        return None
                games["commence_local_str"] = games["commence_time_utc"].apply(_to_local_str)

            _games_cache = games

    if force_reload:
        _trends_cache_by_days = {}
        _latest_team_features_cache = {}
        _team_name_index = {}

    _ = get_trends_df(DEFAULT_SPAN_DAYS)
    _build_latest_team_feature_cache()
    return _games_cache

def get_trends_df(days: int) -> pd.DataFrame:
    global _trends_cache_by_days

    if days in _trends_cache_by_days:
        return _trends_cache_by_days[days]

    span_path = _trend_file_for_days(days)
    if os.path.exists(span_path):
        df = _load_trends_csv(span_path)
        _trends_cache_by_days[days] = df
        return df

    big_path = _trend_file_for_days(365)
    if os.path.exists(big_path):
        big_df = _trends_cache_by_days.get(365)
        if big_df is None:
            big_df = _load_trends_csv(big_path)
            _trends_cache_by_days[365] = big_df

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        if "date_utc" in big_df.columns:
            df = big_df[big_df["date_utc"] >= cutoff].copy()
        else:
            df = big_df.copy()

        _trends_cache_by_days[days] = df
        return df

    if os.path.exists(TRENDS_FALLBACK):
        df = _load_trends_csv(TRENDS_FALLBACK)
        _trends_cache_by_days[days] = df
        return df

    empty = pd.DataFrame(columns=["league", "date_utc", "team"])
    _trends_cache_by_days[days] = empty
    return empty

def _build_latest_team_feature_cache():
    global _latest_team_features_cache, _team_name_index
    if _latest_team_features_cache and _team_name_index:
        return

    trends_df = get_trends_df(365)
    if trends_df is None or trends_df.empty:
        return
    if not all(c in trends_df.columns for c in ["league", "team", "date_utc"]):
        return

    if not pd.api.types.is_datetime64_any_dtype(trends_df["date_utc"]):
        trends_df = trends_df.copy()
        trends_df["date_utc"] = pd.to_datetime(trends_df["date_utc"], utc=True, errors="coerce")

    tdf = trends_df.dropna(subset=["date_utc"]).copy()
    if tdf.empty:
        return

    _team_name_index = {}
    for league, gdf in tdf.groupby("league"):
        m = {}
        for t in gdf["team"].dropna().astype(str).unique().tolist():
            m[_norm_team(t)] = t
        _team_name_index[str(league)] = m

    tdf = tdf.sort_values(["league", "team", "date_utc"])
    latest = tdf.groupby(["league", "team"], as_index=False).tail(1)

    _latest_team_features_cache = {}
    for _, r in latest.iterrows():
        league = r.get("league")
        team = r.get("team")
        if league is None or team is None:
            continue
        _latest_team_features_cache[(str(league), str(team))] = r.to_dict()

def _series_for_two_teams(trends_df: pd.DataFrame, league: str, home_team: str, away_team: str, metric: str):
    if trends_df is None or trends_df.empty:
        return [], [], [], METRICS.get(metric, {}).get("kind", "num")
    if metric not in trends_df.columns:
        return [], [], [], METRICS.get(metric, {}).get("kind", "num")

    home_team_r = _resolve_team_name(league, home_team)
    away_team_r = _resolve_team_name(league, away_team)

    home_df = trends_df[(trends_df["league"] == league) & (trends_df["team"] == home_team_r)].copy()
    away_df = trends_df[(trends_df["league"] == league) & (trends_df["team"] == away_team_r)].copy()

    if "date_utc" not in home_df.columns or "date_utc" not in away_df.columns:
        return [], [], [], METRICS.get(metric, {}).get("kind", "num")

    home_df = home_df.sort_values("date_utc")
    away_df = away_df.sort_values("date_utc")

    def date_key(s):
        return s.date().isoformat()

    home_map = {}
    for _, r in home_df.iterrows():
        ts = r.get("date_utc")
        if pd.isna(ts):
            continue
        home_map[date_key(ts)] = _safe_float(r.get(metric))

    away_map = {}
    for _, r in away_df.iterrows():
        ts = r.get("date_utc")
        if pd.isna(ts):
            continue
        away_map[date_key(ts)] = _safe_float(r.get(metric))

    labels = sorted(set(home_map.keys()) | set(away_map.keys()))
    home_series = [home_map.get(d) for d in labels]
    away_series = [away_map.get(d) for d in labels]
    kind = METRICS.get(metric, {}).get("kind", "num")
    return labels, home_series, away_series, kind

def _game_row(event_id: str):
    df = load_data()
    if df is None or df.empty:
        return None
    row = df[df["event_id"].astype(str) == str(event_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def _devig_probs_from_moneylines(home_ml, away_ml):
    def implied(ml):
        ml = _safe_float(ml)
        if ml is None:
            return None
        if ml > 0:
            return 100.0 / (ml + 100.0)
        return (-ml) / ((-ml) + 100.0)

    ph = implied(home_ml)
    pa = implied(away_ml)
    if ph is None or pa is None:
        return None, None
    s = ph + pa
    if s <= 0:
        return None, None
    return ph / s, pa / s


# ----------------------------
# Odds math helpers
# ----------------------------
def american_to_implied_prob(a):
    a = _safe_float(a)
    if a is None or a == 0:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    return abs(a) / (abs(a) + 100.0)

def clamp01(x):
    if x is None:
        return None
    return max(0.01, min(0.99, float(x)))

def normal_cdf(x, mu, sigma):
    if sigma is None or sigma <= 0:
        return None
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def _fmt_american_display(a):
    a = _safe_float(a)
    if a is None:
        return "—"
    ai = int(round(a))
    return f"+{ai}" if ai > 0 else str(ai)

def _predict_proba_safe(model, row_dict: dict):
    cols = None
    try:
        cols = list(getattr(model, "feature_names_in_", [])) or None
    except Exception:
        cols = None

    if cols is None:
        cols = list(row_dict.keys())

    X = pd.DataFrame([{c: row_dict.get(c) for c in cols}])
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    proba = model.predict_proba(X)
    return float(proba[0, 1])


# ----------------------------
# Model probabilities (moneyline)
# ----------------------------
def model_prob_moneyline(g):
    league = str(g.get("league") or "")
    home_raw = str(g.get("home_team") or "")
    away_raw = str(g.get("away_team") or "")

    _build_latest_team_feature_cache()

    home_team = _resolve_team_name(league, home_raw)
    away_team = _resolve_team_name(league, away_raw)

    h = _latest_team_features_cache.get((league, home_team))
    a = _latest_team_features_cache.get((league, away_team))

    if not h or not a:
        return None, None

    model = ML_MODEL_NBA if league == "NBA" else (ML_MODEL_NFL if league == "NFL" else None)
    if model is None:
        return None, None

    base = {
        "rolling_win_pct_10": _safe_float(h.get("rolling_win_pct_10")),
        "rolling_point_diff_10": _safe_float(h.get("rolling_point_diff_10")),
        "rolling_pf_10": _safe_float(h.get("rolling_pf_10")),
        "rolling_pa_10": _safe_float(h.get("rolling_pa_10")),
        "is_home": 1,
    }
    base_away = {
        "rolling_win_pct_10": _safe_float(a.get("rolling_win_pct_10")),
        "rolling_point_diff_10": _safe_float(a.get("rolling_point_diff_10")),
        "rolling_pf_10": _safe_float(a.get("rolling_pf_10")),
        "rolling_pa_10": _safe_float(a.get("rolling_pa_10")),
        "is_home": 0,
    }

    try:
        p_home_raw = _predict_proba_safe(model, base)
        p_away_raw = _predict_proba_safe(model, base_away)
    except Exception:
        return None, None

    s = p_home_raw + p_away_raw
    if s <= 1e-9:
        return None, None

    p_home = clamp01(p_home_raw / s)
    return p_home, clamp01(1.0 - p_home)

def model_prob_spread(g, side: str):
    home_spread = _safe_float(g.get("spread_home_point"))
    away_spread = _safe_float(g.get("spread_away_point"))
    if home_spread is None and away_spread is None:
        return None

    league = str(g.get("league") or "")

    mu = None
    h_avg = _safe_float(g.get("home_avg_diff"))
    a_avg = _safe_float(g.get("away_avg_diff"))
    if None not in (h_avg, a_avg):
        mu = (h_avg - a_avg)

    if mu is None:
        return None

    sigma = 11.0 if league == "NBA" else 10.0

    if home_spread is None and away_spread is not None:
        home_spread = -away_spread

    thresh_home = -home_spread
    p_home = 1.0 - normal_cdf(thresh_home, mu, sigma)
    p_home = clamp01(p_home)
    return p_home if side == "HOME" else clamp01(1.0 - p_home)

def model_prob_total(g, side: str):
    total = _safe_float(g.get("total_points"))
    if total is None:
        return None

    league = str(g.get("league") or "")

    h_ppg = _safe_float(g.get("home_ppg"))
    h_opp = _safe_float(g.get("home_opp_ppg"))
    a_ppg = _safe_float(g.get("away_ppg"))
    a_opp = _safe_float(g.get("away_opp_ppg"))

    mu = None
    if None not in (h_ppg, h_opp, a_ppg, a_opp):
        mu = (h_ppg + a_ppg + h_opp + a_opp) / 2.0
    if mu is None:
        return None

    sigma = 16.0 if league == "NBA" else 13.0
    p_over = 1.0 - normal_cdf(total, mu, sigma)
    p_over = clamp01(p_over)
    return p_over if side == "OVER" else clamp01(1.0 - p_over)


# ----------------------------
# Legs / bet quality
# ----------------------------
def _finalize_leg(leg: dict) -> dict:
    leg = dict(leg)
    leg["american"] = _safe_float(leg.get("american"))
    leg["displayOdds"] = _fmt_american_display(leg.get("american"))

    p_mkt = american_to_implied_prob(leg.get("american"))
    leg["p_market"] = clamp01(p_mkt) if p_mkt is not None else None

    pm = _safe_float(leg.get("p_model"))
    leg["p_model"] = clamp01(pm) if pm is not None else None

    if leg["p_model"] is not None and leg["p_market"] is not None:
        leg["edge"] = float(leg["p_model"]) - float(leg["p_market"])
    else:
        leg["edge"] = None

    return leg

def build_legs_from_games(df: pd.DataFrame):
    legs = []
    for g in df.to_dict(orient="records"):
        event_id = str(g.get("event_id"))
        league = g.get("league")
        matchup = f"{g.get('away_team')} @ {g.get('home_team')}"

        ph, pa = model_prob_moneyline(g)

        if g.get("home_ml_best") is not None:
            legs.append(_finalize_leg({
                "event_id": event_id, "league": league, "matchup": matchup,
                "market": "Moneyline", "selection": "HOME_ML",
                "label": f"{g.get('home_team')} ML",
                "american": g.get("home_ml_best"),
                "p_model": ph,
                "point": None,
            }))
        if g.get("away_ml_best") is not None:
            legs.append(_finalize_leg({
                "event_id": event_id, "league": league, "matchup": matchup,
                "market": "Moneyline", "selection": "AWAY_ML",
                "label": f"{g.get('away_team')} ML",
                "american": g.get("away_ml_best"),
                "p_model": pa,
                "point": None,
            }))

        if g.get("spread_home_point") is not None and g.get("spread_home_price_best") is not None:
            p = model_prob_spread(g, "HOME")
            legs.append(_finalize_leg({
                "event_id": event_id, "league": league, "matchup": matchup,
                "market": "Spread", "selection": "HOME_SPREAD",
                "label": f"{g.get('home_team')} {g.get('spread_home_point')}",
                "american": g.get("spread_home_price_best"),
                "p_model": p,
                "point": _safe_float(g.get("spread_home_point")),
            }))
        if g.get("spread_away_point") is not None and g.get("spread_away_price_best") is not None:
            p = model_prob_spread(g, "AWAY")
            legs.append(_finalize_leg({
                "event_id": event_id, "league": league, "matchup": matchup,
                "market": "Spread", "selection": "AWAY_SPREAD",
                "label": f"{g.get('away_team')} {g.get('spread_away_point')}",
                "american": g.get("spread_away_price_best"),
                "p_model": p,
                "point": _safe_float(g.get("spread_away_point")),
            }))

        if g.get("total_points") is not None and g.get("total_over_price_best") is not None:
            p = model_prob_total(g, "OVER")
            legs.append(_finalize_leg({
                "event_id": event_id, "league": league, "matchup": matchup,
                "market": "Total", "selection": "OVER",
                "label": f"Over {g.get('total_points')}",
                "american": g.get("total_over_price_best"),
                "p_model": p,
                "point": _safe_float(g.get("total_points")),
            }))
        if g.get("total_points") is not None and g.get("total_under_price_best") is not None:
            p = model_prob_total(g, "UNDER")
            legs.append(_finalize_leg({
                "event_id": event_id, "league": league, "matchup": matchup,
                "market": "Total", "selection": "UNDER",
                "label": f"Under {g.get('total_points')}",
                "american": g.get("total_under_price_best"),
                "p_model": p,
                "point": _safe_float(g.get("total_points")),
            }))

    out = []
    for l in legs:
        if l.get("american") is None:
            continue
        if l.get("p_market") is None:
            continue
        out.append(l)
    return out

def bet_quality_for_game(g):
    df = pd.DataFrame([g])
    legs = build_legs_from_games(df)

    def _key(x):
        e = x.get("edge")
        return -1e18 if e is None else float(e)

    legs.sort(key=_key, reverse=True)
    return legs

def find_leg(event_id: str, market: str, selection: str):
    g = _game_row(event_id)
    if g is None:
        return None
    legs = bet_quality_for_game(g)
    for l in legs:
        if (str(l.get("event_id")) == str(event_id)
            and str(l.get("market")) == str(market)
            and str(l.get("selection")) == str(selection)):
            return l
    return None


# ----------------------------
# Parlay placement API (coins)
# ----------------------------
@app.post("/api/parlays/place")
@login_required
def api_place_parlay():
    data = request.get_json(silent=True) or {}
    legs = data.get("legs") or []
    stake = data.get("stake")

    try:
        stake = int(stake)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid stake"}), 400

    if stake <= 0:
        return jsonify({"ok": False, "error": "Stake must be > 0"}), 400
    if not isinstance(legs, list) or len(legs) < 2:
        return jsonify({"ok": False, "error": "Parlay needs at least 2 legs"}), 400
    if len(legs) > 12:
        return jsonify({"ok": False, "error": "Max 12 legs"}), 400

    dec = 1.0
    leg_rows = []

    for leg in legs:
        try:
            american = int(float(leg.get("american")))
        except Exception:
            return jsonify({"ok": False, "error": "Each leg needs valid american odds"}), 400

        event_id = str(leg.get("event_id") or "").strip()
        league = str(leg.get("league") or "").strip()
        market = str(leg.get("market") or "").strip()
        selection = str(leg.get("selection") or "").strip()
        label = str(leg.get("label") or "").strip()
        matchup = str(leg.get("matchup") or "").strip()

        if not all([event_id, league, market, selection, label, matchup]):
            return jsonify({"ok": False, "error": "Leg missing required fields"}), 400

        point = leg.get("point", None)
        try:
            point = float(point) if point is not None else None
        except Exception:
            point = None

        if american == 0:
            return jsonify({"ok": False, "error": "Invalid american odds 0"}), 400
        if american > 0:
            dec *= (1.0 + american / 100.0)
        else:
            dec *= (1.0 + 100.0 / abs(american))

        leg_rows.append(ParlayLeg(
            event_id=event_id,
            league=league,
            market=market,
            selection=selection,
            label=label,
            matchup=matchup,
            american=american,
            point=point,
        ))

    amer_total = None
    if dec > 1:
        if dec >= 2:
            amer_total = int(round((dec - 1) * 100))
        else:
            amer_total = int(-round(100 / (dec - 1)))

    try:
        u_q = db.session.query(User).filter(User.id == current_user.id)
        try:
            u = u_q.with_for_update().one()
        except Exception:
            u = u_q.one()

        if int(u.balance or 0) < stake:
            return jsonify({"ok": False, "error": "Not enough coins"}), 400

        u.balance = int(u.balance or 0) - stake

        p = Parlay(
            user_id=u.id,
            stake=stake,
            decimal_odds=float(dec),
            american_odds=amer_total,
            status="PENDING",
            payout=0,
        )
        db.session.add(p)
        db.session.flush()

        for lr in leg_rows:
            lr.parlay_id = p.id
            db.session.add(lr)

        db.session.commit()
        db.session.refresh(u)
        db.session.refresh(p)

        return jsonify({
            "ok": True,
            "parlay_id": p.id,
            "new_balance": int(u.balance or 0),
            "decimal_odds": round(float(dec), 4),
            "american_odds": amer_total,
            "status": p.status,
        })
    except IntegrityError:
        db.session.rollback()
        return jsonify({"ok": False, "error": "Database error (integrity)"}), 500
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": f"Server error: {e}"}), 500


@app.get("/api/parlays/mine")
@login_required
def api_my_parlays():
    parlays = (
        db.session.query(Parlay)
        .filter(Parlay.user_id == current_user.id)
        .order_by(Parlay.placed_at.desc())
        .limit(50)
        .all()
    )

    out = []
    for p in parlays:
        legs = (
            db.session.query(ParlayLeg)
            .filter(ParlayLeg.parlay_id == p.id)
            .all()
        )
        out.append({
            "id": p.id,
            "stake": p.stake,
            "decimal_odds": p.decimal_odds,
            "american_odds": p.american_odds,
            "status": p.status,
            "payout": p.payout,
            "placed_at": p.placed_at.isoformat() if p.placed_at else None,
            "settled_at": p.settled_at.isoformat() if p.settled_at else None,
            "legs": [{
                "event_id": l.event_id,
                "league": l.league,
                "market": l.market,
                "selection": l.selection,
                "label": l.label,
                "matchup": l.matchup,
                "american": l.american,
                "point": l.point,
            } for l in legs]
        })
    return jsonify({"ok": True, "parlays": out})


# ----------------------------
# Game APIs
# ----------------------------
@app.get("/api/game/<event_id>/trends")
def api_game_trends(event_id: str):
    try:
        g = _game_row(event_id)
        if g is None:
            return jsonify({"ok": False, "error": "not_found"}), 404

        metric = (request.args.get("metric") or "").strip()
        if metric not in METRICS:
            metric = "rolling_win_pct_10"

        try:
            span = int(request.args.get("span") or DEFAULT_SPAN_DAYS)
        except Exception:
            span = DEFAULT_SPAN_DAYS
        if span not in SPAN_CHOICES:
            span = DEFAULT_SPAN_DAYS

        trends_df = get_trends_df(span)

        labels, home_series, away_series, kind = _series_for_two_teams(
            trends_df,
            str(g.get("league") or ""),
            str(g.get("home_team") or ""),
            str(g.get("away_team") or ""),
            metric,
        )

        return jsonify({
            "ok": True,
            "event_id": str(event_id),
            "metric": metric,
            "span": span,
            "kind": kind or "num",
            "labels": labels or [],
            "home": home_series or [],
            "away": away_series or [],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": "server_error", "detail": str(e)}), 500


@app.get("/api/game/<event_id>/bet_quality")
def api_game_bet_quality(event_id: str):
    g = _game_row(event_id)
    if g is None:
        return jsonify({"error": "not_found"}), 404
    legs = bet_quality_for_game(g)
    return jsonify({"event_id": str(event_id), "legs": legs, "note": "Informational only — not betting advice."})


@app.get("/api/leg")
def api_leg():
    event_id = request.args.get("event_id")
    market = request.args.get("market")
    selection = request.args.get("selection")
    if not event_id or not market or not selection:
        return jsonify({"error": "missing_params", "need": ["event_id", "market", "selection"]}), 400
    leg = find_leg(event_id, market, selection)
    if leg is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"leg": leg})


@app.get("/api/me")
def api_me():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "username": current_user.username,
        "balance": int(current_user.balance or 0),
    })


# ----------------------------
# Auth routes
# ----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("main"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
            return render_template("signup.html", title="Sign up")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("signup.html", title="Sign up")

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("That username is already taken.", "danger")
            return render_template("signup.html", title="Sign up")

        u = User(username=username, balance=1000)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()

        login_user(u)
        flash("Account created! You have 1000 coins.", "success")
        return redirect(url_for("main"))

    return render_template("signup.html", title="Sign up")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main"))

    next_url = request.args.get("next") or url_for("main")

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        try:
            u = User.query.filter_by(username=username).first()
        except Exception:
            db.session.rollback()
            raise

        if not u or not u.check_password(password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html", title="Login", next_url=next_url)

        login_user(u)
        flash("Logged in.", "success")
        return redirect(next_url)

    return render_template("login.html", title="Login", next_url=next_url)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("main"))


_best_cache = {"t": 0.0, "rows": []}
BEST_CACHE_SECONDS = int(os.environ.get("BEST_CACHE_SECONDS", "60"))  # cache best picks for 60s
BEST_MAX_GAMES = int(os.environ.get("BEST_MAX_GAMES", "150"))         # cap work

# ----------------------------
# Routes (site)
# ----------------------------
@app.route("/")
def main():
    load_data()
    df = _games_cache

    nba_games = []
    nfl_games = []
    if df is not None and not df.empty:
        nba_games = df[df["league"] == "NBA"].to_dict(orient="records")
        nfl_games = df[df["league"] == "NFL"].to_dict(orient="records")

    now_local = datetime.now(LOCAL_TZ)
    today_str = now_local.strftime("%a %b %d, %Y")
    loaded_at = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")

    # ---- FAST best picks (cached + bounded) ----
    best = []
    try:
        now_ts = time.time()
        if (now_ts - float(_best_cache.get("t", 0.0))) < BEST_CACHE_SECONDS:
            best = _best_cache.get("rows", []) or []
        else:
            best = []

            if df is not None and not df.empty:
                df2 = df

                # Prefer upcoming games first; this also avoids scoring old rows if present
                if "commence_time_utc" in df2.columns:
                    # ensure datetime
                    try:
                        ct = pd.to_datetime(df2["commence_time_utc"], utc=True, errors="coerce")
                        df2 = df2.assign(_ct=ct)
                        df2 = df2.sort_values("_ct", ascending=True)
                    except Exception:
                        pass

                # Hard cap to keep work predictable
                df2 = df2.head(BEST_MAX_GAMES)

                legs = build_legs_from_games(df2)
                legs = [l for l in legs if l.get("edge") is not None]
                legs.sort(key=lambda x: float(x["edge"]), reverse=True)
                best = legs[:12]

            _best_cache["t"] = now_ts
            _best_cache["rows"] = best
    except Exception:
        best = []

    return render_template(
        "main.html",
        title="The Odds Den",
        nba_games=nba_games,
        nfl_games=nfl_games,
        today_str=today_str,
        loaded_at=loaded_at,
        config={"LOCAL_TZ": str(LOCAL_TZ)},
        best_picks_json=json.dumps(best),
    )

@app.route("/game/<event_id>")
def game(event_id: str):
    g = _game_row(event_id)
    if g is None:
        abort(404, description="Game not found in CSV")

    home_prob, away_prob = _devig_probs_from_moneylines(g.get("home_ml_best"), g.get("away_ml_best"))

    span_days = int(request.args.get("span", DEFAULT_SPAN_DAYS))
    span_days = span_days if span_days in SPAN_CHOICES else DEFAULT_SPAN_DAYS

    metric_a = request.args.get("m1", "rolling_win_pct_10")
    metric_b = request.args.get("m2", "point_diff")
    if metric_a not in METRICS:
        metric_a = "rolling_win_pct_10"
    if metric_b not in METRICS:
        metric_b = "point_diff"

    trends_df = get_trends_df(span_days)

    labels_a, home_a, away_a, kind_a = _series_for_two_teams(trends_df, g["league"], g["home_team"], g["away_team"], metric_a)
    labels_b, home_b, away_b, kind_b = _series_for_two_teams(trends_df, g["league"], g["home_team"], g["away_team"], metric_b)

    game_legs = bet_quality_for_game(g)
    game_legs_json = json.dumps(game_legs)

    return render_template(
        "game.html",
        title=f"{g['away_team']} @ {g['home_team']} — The Odds Den",
        g=g,
        home_prob=home_prob,
        away_prob=away_prob,
        span_days=span_days,
        span_choices=SPAN_CHOICES,
        metrics=METRICS,
        metric_a=metric_a,
        metric_b=metric_b,
        labels_a=labels_a,
        home_a=home_a,
        away_a=away_a,
        kind_a=kind_a,
        labels_b=labels_b,
        home_b=home_b,
        away_b=away_b,
        kind_b=kind_b,
        game_legs_json=game_legs_json,
    )


@app.route("/reload", endpoint="reload_data")
def reload_data():
    load_data(force_reload=True)
    return "Reloaded games + trends CSV into memory. Go back to /"


# ----------------------------
# Startup: ensure schema + (optionally) start worker
# ----------------------------
with app.app_context():
    try:
        ensure_schema()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass

# If you want the worker to start even before first request (useful for local runs),
# set START_WORKER_ON_BOOT=1
if os.environ.get("START_WORKER_ON_BOOT", "0") == "1" and os.environ.get("RUN_WORKER_IN_WEB", "1") == "1":
    try:
        start_worker_if_needed()
    except Exception:
        pass

# Kick off odds scraper on boot (default on) so odds refresh immediately
if os.environ.get("START_SCRAPER_ON_BOOT", "1") == "1" and os.environ.get("RUN_SCRAPER_IN_WEB", "1") == "1":
    try:
        start_scraper_if_needed()
    except Exception:
        pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
