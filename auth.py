# auth.py
from __future__ import annotations

import os
from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user

from models import db, User

auth_bp = Blueprint("auth", __name__)

STARTING_BALANCE = float(os.environ.get("STARTING_BALANCE", "1000"))

@auth_bp.post("/api/auth/register")
def register():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password or len(password) < 6:
        return jsonify({"error": "invalid_input"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email_taken"}), 409

    u = User(email=email, balance=STARTING_BALANCE)
    u.set_password(password)

    db.session.add(u)
    db.session.commit()

    login_user(u)
    return jsonify({"ok": True, "user": {"id": u.id, "email": u.email, "balance": u.balance}})

@auth_bp.post("/api/auth/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    u = User.query.filter_by(email=email).first()
    if not u or not u.check_password(password):
        return jsonify({"error": "bad_credentials"}), 401

    login_user(u)
    return jsonify({"ok": True, "user": {"id": u.id, "email": u.email, "balance": u.balance}})

@auth_bp.post("/api/auth/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"ok": True})

@auth_bp.get("/api/me")
def me():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "user": {"id": current_user.id, "email": current_user.email, "balance": current_user.balance}
    })
