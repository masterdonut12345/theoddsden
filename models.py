from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    balance = db.Column(db.Integer, nullable=False, default=1000)  # coins

    def set_password(self, pw: str) -> None:
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self.password_hash, pw)


class Parlay(db.Model):
    __tablename__ = "parlays"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user = db.relationship("User", backref="parlays")

    stake = db.Column(db.Integer, nullable=False)               # coins
    decimal_odds = db.Column(db.Float, nullable=True)
    american_odds = db.Column(db.Integer, nullable=True)

    status = db.Column(db.String(16), nullable=False, default="PENDING")  # PENDING/WON/LOST/PUSH/VOID
    payout = db.Column(db.Integer, nullable=False, default=0)             # coins actually credited

    placed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    settled_at = db.Column(db.DateTime, nullable=True)

    legs = db.relationship("ParlayLeg", cascade="all, delete-orphan", backref="parlay")


class ParlayLeg(db.Model):
    __tablename__ = "parlay_legs"
    id = db.Column(db.Integer, primary_key=True)

    parlay_id = db.Column(db.Integer, db.ForeignKey("parlays.id"), nullable=False, index=True)

    event_id = db.Column(db.String(64), nullable=False, index=True)  # Odds API event id (string)
    league = db.Column(db.String(8), nullable=False)                 # NBA/NFL

    market = db.Column(db.String(16), nullable=False)                # Moneyline/Spread/Total
    selection = db.Column(db.String(32), nullable=False)             # HOME_ML, AWAY_SPREAD, OVER, etc.

    label = db.Column(db.String(200), nullable=False)
    matchup = db.Column(db.String(200), nullable=False)

    american = db.Column(db.Integer, nullable=False)                 # e.g. -110, +120
    point = db.Column(db.Float, nullable=True)                       # spread point or total points

class EventResult(db.Model):
    __tablename__ = "event_results"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    league = db.Column(db.String(8), nullable=True)

    # Final score
    home_score = db.Column(db.Integer, nullable=True)
    away_score = db.Column(db.Integer, nullable=True)

    # Winner: "HOME" or "AWAY"
    winner = db.Column(db.String(8), nullable=True)

    # Mark final
    final = db.Column(db.Boolean, nullable=False, default=False)

    # When we recorded it
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

def _leg_won(leg: ParlayLeg, r: EventResult) -> bool | None:
    """
    Returns:
      True  -> leg won
      False -> leg lost
      None  -> can't determine yet (missing data)
    """
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
        # Your labels store home spread like -3.5, away spread like +3.5
        # Win condition: (team_score + spread) > opp_score  (push counts as loss here; adjust if you want push=refund)
        if sel == "HOME_SPREAD":
            return (diff_home + float(leg.point)) > 0.0
        if sel == "AWAY_SPREAD":
            # away diff is -diff_home
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
    # simple payout: stake * decimal_odds (rounded)
    try:
        return int(round(float(stake) * float(decimal_odds)))
    except Exception:
        return 0


def settle_parlays_once(limit: int = 100) -> int:
    """
    Settles up to `limit` pending parlays whose legs all have final EventResult rows.
    Safe for Postgres + multiple processes using an advisory lock.
    Returns number of parlays settled.
    """
    # Postgres advisory lock to prevent double settlement across worker instances
    # (If not Postgres, this will fail and we just proceed without it.)
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

            # Load results for all legs
            event_ids = list({str(l.event_id) for l in legs})
            results = (
                db.session.query(EventResult)
                .filter(EventResult.event_id.in_(event_ids))
                .all()
            )
            res_by_event = {r.event_id: r for r in results}

            # If any leg missing final result -> skip for now
            outcomes: list[bool] = []
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

            # Settle
            all_won = all(outcomes)
            p.settled_at = datetime.utcnow()

            if all_won:
                p.status = "WON"
                p.payout = _parlay_payout(p.stake, p.decimal_odds or 0.0)

                # credit user
                u = db.session.query(User).filter(User.id == p.user_id).one()
                u.balance += int(p.payout)
            else:
                p.status = "LOST"
                p.payout = 0

            settled += 1

        db.session.flush()
        return settled

    finally:
        # release advisory lock if we got it
        if got_lock:
            try:
                db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": 987654321})
            except Exception:
                pass
