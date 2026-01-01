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
# Models (match your provided models, plus last_claimed_at)
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
# Admin token helpers (same style as /admin/init_db)
# ----------------------------
def _require_admin_token() -> bool:
    token = os.environ.get("ADMIN_TOKEN", "")
    supplied = (request.args.get("token") or "").strip()
    return bool(token) and supplied == token

def _admin_unauth():
    return jsonify({"ok": False, "error": "unauthorized"}), 401


# ----------------------------
# Admin endpoints
# ----------------------------
@app.get("/admin/init_db")
def admin_init_db():
    # creates tables only; DOES NOT add new columns to existing tables
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
    """
    Safe migration endpoint:
    - create_all()
    - add missing columns we rely on (like users.last_claimed_at)
    """
    if not _require_admin_token():
        return _admin_unauth()

    try:
        with app.app_context():
            db.create_all()

            # Add columns if missing (Postgres supports IF NOT EXISTS; SQLite will ignore errors via try/except)
            # last_claimed_at is REQUIRED for daily claim.
            try:
                db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_claimed_at TIMESTAMP NULL"))
            except Exception:
                # SQLite does not support IF NOT EXISTS on older versions; try plain ALTER, ignore if it already exists.
                try:
                    db.session.execute(text("ALTER TABLE users ADD COLUMN last_claimed_at TIMESTAMP NULL"))
                except Exception:
                    pass

            db.session.commit()

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

        # delete dependent rows explicitly (safe even if cascades not configured)
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

        # best effort row lock in Postgres
        try:
            u = db.session.query(User).filter(User.id == u.id).with_for_update().one()
        except Exception:
            pass

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
# Background parlay settler (runs inside web process)
# ----------------------------
_settler_thread = None
_settler_stop = threading.Event()
_settler_lock = threading.Lock()

def _settler_loop(interval_seconds: int):
    while not _settler_stop.is_set():
        try:
            with app.app_context():
                n = settle_parlays_once(limit=200)
                if n:
                    db.session.commit()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass

        # sleep in chunks so stop can interrupt quickly
        for _ in range(max(1, interval_seconds)):
            if _settler_stop.is_set():
                break
            time.sleep(1)

def start_settler_if_needed():
    global _settler_thread
    if _settler_thread is not None and _settler_thread.is_alive():
        return
    with _settler_lock:
        if _settler_thread is not None and _settler_thread.is_alive():
            return
        interval = int(os.environ.get("SETTLE_INTERVAL_SECONDS", "60"))
        _settler_stop.clear()
        _settler_thread = threading.Thread(
            target=_settler_loop,
            args=(interval,),
            daemon=True,
            name="parlay-settler",
        )
        _settler_thread.start()

import atexit
@atexit.register
def _stop_settler():
    _settler_stop.set()


# ----------------------------
# Background odds scraper worker (runs odds_scraper.py)
# ----------------------------
SCRAPE_ENABLED = os.environ.get("SCRAPE_ENABLED", "1") == "1"
ODDS_SCRAPER_PATH = os.environ.get("ODDS_SCRAPER_PATH", "odd_scraper.py")
SCRAPE_OUTDIR = os.environ.get("OUTDIR", ".")  # odds_scraper.py already reads OUTDIR; pass too
SCRAPE_TRENDS_LOCAL_HOUR = int(os.environ.get("SCRAPE_TRENDS_LOCAL_HOUR", "6"))   # 6 AM local
SCRAPE_TRENDS_LOCAL_MINUTE = int(os.environ.get("SCRAPE_TRENDS_LOCAL_MINUTE", "15"))
SCRAPE_GAMES_INTERVAL_MIN = int(os.environ.get("SCRAPE_GAMES_INTERVAL_MIN", "30"))  # refresh odds frequently, but not crazy
SCRAPE_GAMES_MIN_GAP_SECONDS = int(os.environ.get("SCRAPE_GAMES_MIN_GAP_SECONDS", "600"))  # hard floor (10 min)
SCRAPE_SUBPROCESS_TIMEOUT = int(os.environ.get("SCRAPE_SUBPROCESS_TIMEOUT", "1800"))  # 30 min
SCRAPE_LOG_STDOUT = os.environ.get("SCRAPE_LOG_STDOUT", "0") == "1"

# Use advisory locks so multiple Render web instances don't double-run scrapes.
LOCK_KEY_TRENDS = int(os.environ.get("SCRAPE_LOCK_KEY_TRENDS", "314159265"))
LOCK_KEY_GAMES  = int(os.environ.get("SCRAPE_LOCK_KEY_GAMES",  "271828182"))

_scraper_thread = None
_scraper_stop = threading.Event()
_scraper_lock = threading.Lock()

_scrape_state = {
    "trends": {"last_started_utc": None, "last_finished_utc": None, "last_ok": None, "last_error": None},
    "games":  {"last_started_utc": None, "last_finished_utc": None, "last_ok": None, "last_error": None},
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

def _run_odds_scraper(mode: str) -> bool:
    """
    Runs odds_scraper.py in a subprocess.
    - mode='trends': builds 7/30/180/365/730 spans (max 730)
    - mode='games' : builds today's/upcoming slate CSV + odds history
    """
    if not os.path.exists(ODDS_SCRAPER_PATH):
        raise RuntimeError(f"ODDS_SCRAPER_PATH not found: {ODDS_SCRAPER_PATH}")

    env = dict(os.environ)
    env["OUTDIR"] = SCRAPE_OUTDIR  # align with odds_scraper.py defaults

    if mode == "trends":
        cmd = [
            "python3",
            ODDS_SCRAPER_PATH,
            "--mode", "trends",
            "--outdir", SCRAPE_OUTDIR,
            "--days-list", "7", "30", "180", "365", "730",
            "--leagues", "NBA", "NFL",
            "--sleep", os.environ.get("TRENDS_SLEEP", "0.15"),
        ]
    elif mode == "games":
        cmd = [
            "python3",
            ODDS_SCRAPER_PATH,
            "--mode", "games",
            "--outdir", SCRAPE_OUTDIR,
        ]
    else:
        raise ValueError("mode must be 'trends' or 'games'")

    started = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=SCRAPE_SUBPROCESS_TIMEOUT,
    )
    dur = time.time() - started

    if SCRAPE_LOG_STDOUT:
        log.warning("[scraper:%s] stdout:\n%s", mode, (proc.stdout or "").strip())
        log.warning("[scraper:%s] stderr:\n%s", mode, (proc.stderr or "").strip())
    else:
        if proc.returncode != 0:
            # only dump on errors if not logging everything
            log.warning("[scraper:%s] stderr:\n%s", mode, (proc.stderr or "").strip())

    if proc.returncode != 0:
        raise RuntimeError(f"odds_scraper.py {mode} failed (code={proc.returncode}, dur={dur:.1f}s)")

    return True

def _next_trends_run_local(now_local: datetime) -> datetime:
    target = now_local.replace(
        hour=SCRAPE_TRENDS_LOCAL_HOUR,
        minute=SCRAPE_TRENDS_LOCAL_MINUTE,
        second=0,
        microsecond=0,
    )
    if target <= now_local:
        target = target + timedelta(days=1)
    return target

def _scraper_loop():
    """
    Simple scheduler:
    - trends: 1x/day at configured local time
    - games: every SCRAPE_GAMES_INTERVAL_MIN minutes, with a min gap
    Uses PG advisory locks so only one instance actually executes each job.
    """
    last_games_run_ts = 0.0
    next_trends_local = None

    while not _scraper_stop.is_set():
        try:
            now_utc = datetime.now(timezone.utc)
            now_local = now_utc.astimezone(LOCAL_TZ)

            if next_trends_local is None:
                next_trends_local = _next_trends_run_local(now_local)

            # ---- trends daily ----
            if now_local >= next_trends_local:
                with app.app_context():
                    got = _pg_try_lock(LOCK_KEY_TRENDS)
                    if got:
                        try:
                            _scrape_state["trends"]["last_started_utc"] = _utc_now_iso()
                            _scrape_state["trends"]["last_error"] = None
                            ok = _run_odds_scraper("trends")
                            _scrape_state["trends"]["last_ok"] = bool(ok)
                            _scrape_state["trends"]["last_finished_utc"] = _utc_now_iso()
                            # Reload caches if files changed
                            load_data(force_reload=True)
                        except Exception as e:
                            _scrape_state["trends"]["last_ok"] = False
                            _scrape_state["trends"]["last_error"] = str(e)
                            _scrape_state["trends"]["last_finished_utc"] = _utc_now_iso()
                        finally:
                            _pg_unlock(LOCK_KEY_TRENDS)
                    # schedule next day regardless (even if we didn't get lock)
                    next_trends_local = _next_trends_run_local(now_local)

            # ---- games periodic ----
            interval_s = max(60, int(SCRAPE_GAMES_INTERVAL_MIN) * 60)
            min_gap_s = max(60, int(SCRAPE_GAMES_MIN_GAP_SECONDS))
            now_ts = time.time()

            should_games = (now_ts - last_games_run_ts) >= interval_s
            if should_games and (now_ts - last_games_run_ts) >= min_gap_s:
                with app.app_context():
                    got = _pg_try_lock(LOCK_KEY_GAMES)
                    if got:
                        try:
                            _scrape_state["games"]["last_started_utc"] = _utc_now_iso()
                            _scrape_state["games"]["last_error"] = None
                            ok = _run_odds_scraper("games")
                            _scrape_state["games"]["last_ok"] = bool(ok)
                            _scrape_state["games"]["last_finished_utc"] = _utc_now_iso()
                            last_games_run_ts = time.time()
                            load_data(force_reload=True)
                        except Exception as e:
                            _scrape_state["games"]["last_ok"] = False
                            _scrape_state["games"]["last_error"] = str(e)
                            _scrape_state["games"]["last_finished_utc"] = _utc_now_iso()
                            last_games_run_ts = time.time()
                        finally:
                            _pg_unlock(LOCK_KEY_GAMES)
                    else:
                        # Another instance ran it; don't hammer lock
                        last_games_run_ts = time.time()

        except Exception:
            # keep loop alive no matter what
            pass

        # sleep in small chunks so stop is responsive
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

@app.get("/admin/scrape_status")
def admin_scrape_status():
    if not _require_admin_token():
        return _admin_unauth()
    return jsonify({
        "ok": True,
        "enabled": SCRAPE_ENABLED,
        "local_tz": str(LOCAL_TZ),
        "trends_daily_at_local": f"{SCRAPE_TRENDS_LOCAL_HOUR:02d}:{SCRAPE_TRENDS_LOCAL_MINUTE:02d}",
        "games_interval_min": SCRAPE_GAMES_INTERVAL_MIN,
        "state": _scrape_state,
        "paths": {
            "ODDS_SCRAPER_PATH": ODDS_SCRAPER_PATH,
            "OUTDIR": SCRAPE_OUTDIR,
            "DATA_CSV": DATA_GAMES,
            "TRENDS_DIR": TRENDS_DIR,
        }
    })

@app.get("/admin/scrape_now")
def admin_scrape_now():
    if not _require_admin_token():
        return _admin_unauth()

    mode = (request.args.get("mode") or "").strip().lower()
    if mode not in ("trends", "games"):
        return jsonify({"ok": False, "error": "invalid_mode", "allowed": ["trends", "games"]}), 400

    key = LOCK_KEY_TRENDS if mode == "trends" else LOCK_KEY_GAMES
    try:
        with app.app_context():
            got = _pg_try_lock(key)
            if not got:
                return jsonify({"ok": False, "error": "locked", "message": "Another instance is running this job."}), 409
            try:
                _scrape_state[mode]["last_started_utc"] = _utc_now_iso()
                _scrape_state[mode]["last_error"] = None
                ok = _run_odds_scraper(mode)
                _scrape_state[mode]["last_ok"] = bool(ok)
                _scrape_state[mode]["last_finished_utc"] = _utc_now_iso()
                load_data(force_reload=True)
                return jsonify({"ok": True, "mode": mode, "reloaded": True})
            finally:
                _pg_unlock(key)
    except Exception as e:
        _scrape_state[mode]["last_ok"] = False
        _scrape_state[mode]["last_error"] = str(e)
        _scrape_state[mode]["last_finished_utc"] = _utc_now_iso()
        return jsonify({"ok": False, "mode": mode, "error": "run_failed", "detail": str(e)}), 500


# ----------------------------
# Leaderboard
# ----------------------------
@app.get("/leaderboard")
def leaderboard_page():
    # subquery: counts per user
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

    # template-friendly list of dicts
    top_users = []
    for (u, parlays_placed, wins, settled) in rows:
        hit_pct = None
        if settled and int(settled) > 0:
            hit_pct = (float(wins) / float(settled)) * 100.0
        top_users.append({
            "username": u.username,
            "balance": int(u.balance or 0),
            "parlays_placed": int(parlays_placed or 0),
            "hit_pct": hit_pct,  # float or None
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
    # default: 250 coins; override with env
    daily_amount = int(os.environ.get("DAILY_CLAIM_COINS", "250"))
    now = datetime.now(timezone.utc)

    try:
        # lock row if possible
        u = current_user
        u_db = User.query.filter_by(id=u.id).first()
        if not u_db:
            return jsonify({"ok": False, "error": "user_not_found"}), 404

        try:
            u_db = db.session.query(User).filter(User.id == u_db.id).with_for_update().one()
        except Exception:
            pass

        last = u_db.last_claimed_at
        if last is not None:
            # treat "daily" as UTC day boundary
            if last.replace(tzinfo=timezone.utc).date() == now.date():
                return jsonify({
                    "ok": False,
                    "error": "already_claimed",
                    "message": "You already claimed today.",
                    "balance": int(u_db.balance or 0),
                }), 400

        u_db.balance = int(u_db.balance or 0) + daily_amount
        u_db.last_claimed_at = now.replace(tzinfo=None)  # store naive utc
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
    loaded_at = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

    best = []
    try:
        if df is not None and not df.empty:
            legs = build_legs_from_games(df)
            legs = [l for l in legs if l.get("edge") is not None]
            legs.sort(key=lambda x: float(x["edge"]), reverse=True)
            best = legs[:12]
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
# Ensure background workers start
# ----------------------------
@app.before_request
def _ensure_workers_started():
    # Starts once when the first HTTP request hits this process
    start_settler_if_needed()
    start_scraper_if_needed()


if __name__ == "__main__":
    # Local dev: start workers immediately too
    start_settler_if_needed()
    start_scraper_if_needed()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
