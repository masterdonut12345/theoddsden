from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from models import db, Parlay, ParlayLeg

parlays_api = Blueprint("parlays_api", __name__)

def american_to_decimal(a: int) -> float:
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))

def decimal_to_american(dec: float) -> int | None:
    if not dec or dec <= 1:
        return None
    if dec >= 2:
        return int(round((dec - 1) * 100))
    return int(-round(100 / (dec - 1)))

@parlays_api.post("/api/parlays/place")
@login_required
def place_parlay():
    data = request.get_json(silent=True) or {}
    stake = data.get("stake")
    legs = data.get("legs") or []

    # ---- validate
    try:
        stake = int(stake)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid stake"}), 400

    if stake <= 0:
        return jsonify({"ok": False, "error": "Stake must be > 0"}), 400

    if not isinstance(legs, list) or len(legs) < 2:
        return jsonify({"ok": False, "error": "Parlay needs at least 2 legs"}), 400

    # ---- compute odds from leg americans
    dec = 1.0
    leg_rows: list[ParlayLeg] = []

    for leg in legs:
        try:
            american = int(leg.get("american"))
        except Exception:
            return jsonify({"ok": False, "error": "Leg missing valid american odds"}), 400

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

        dec *= american_to_decimal(american)
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

    amer_total = decimal_to_american(dec)

    # ---- atomic-ish balance update + create parlay
    with db.session.begin():
        # refresh current_user row and lock by updating inside txn
        u = db.session.get(type(current_user), current_user.id)
        if u.balance < stake:
            return jsonify({"ok": False, "error": "Not enough coins"}), 400

        u.balance -= stake

        p = Parlay(
            user_id=u.id,
            stake=stake,
            decimal_odds=dec,
            american_odds=amer_total,
            status="PENDING",
            placed_at=datetime.utcnow(),
        )
        db.session.add(p)
        db.session.flush()  # get p.id

        for lr in leg_rows:
            lr.parlay_id = p.id
            db.session.add(lr)

    return jsonify({
        "ok": True,
        "parlay_id": p.id,
        "new_balance": current_user.balance,
        "decimal_odds": round(dec, 4),
        "american_odds": amer_total,
    })
