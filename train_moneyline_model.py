#!/usr/bin/env python3
import sys
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


FEATURES = [
    # core 10-game rolls
    "rolling_win_pct_10",
    "rolling_point_diff_10",
    "rolling_pf_10",
    "rolling_pa_10",

    # extra rolling windows
    "rolling_win_pct_3", "rolling_point_diff_3", "rolling_pf_3", "rolling_pa_3",
    "rolling_win_pct_5", "rolling_point_diff_5", "rolling_pf_5", "rolling_pa_5",
    "rolling_win_pct_20", "rolling_point_diff_20", "rolling_pf_20", "rolling_pa_20",

    # cumulative / momentum
    "cum_wins",
    "cum_losses",
    "game_num",
    "streak_signed",
    "rest_days",

    # EMA smoothers
    "ema_point_diff_10",
    "ema_pf_10",
    "ema_pa_10",

    # derived from homeAway (NOT in CSV)
    "is_home",
]


def _require_cols(df: pd.DataFrame, cols: list[str], league: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{league}] Missing required columns in trends CSV: {missing}")


def _time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict time split: earliest (1-test_frac) for train, latest test_frac for test.
    """
    df = df.sort_values("date_utc").copy()
    n = len(df)
    if n < 200:
        # still works, but warning helps you understand tiny-data weirdness
        print(f"[split] WARNING: only {n} rows; scores may be noisy.")
    cut = int((1.0 - test_frac) * n)
    cut = max(1, min(n - 1, cut))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_one(df: pd.DataFrame, league: str, out_path: str) -> None:
    d = df[df["league"] == league].copy()

    # must have date_utc for time split + no-leak checks
    required_in_csv = ["win", "homeAway", "date_utc"] + [c for c in FEATURES if c != "is_home"]
    _require_cols(d, required_in_csv, league)

    # parse date_utc
    d["date_utc"] = pd.to_datetime(d["date_utc"], utc=True, errors="coerce")

    # create is_home from homeAway
    d["is_home"] = (d["homeAway"] == "home").astype(int)

    # coerce numeric feature columns
    for c in FEATURES:
        if c == "is_home":
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["win"] = pd.to_numeric(d["win"], errors="coerce")

    # drop NA rows needed for training (date + features + label)
    d = d.dropna(subset=["date_utc"] + FEATURES + ["win"]).copy()

    # IMPORTANT: If your features are truly pre-game, the first game per team will have
    # lots of NaNs; dropping them is expected.
    if d.empty:
        raise SystemExit(f"[{league}] No usable rows after dropping NA. Check trends generation.")

    # time-based split
    train_df, test_df = _time_split(d, test_frac=0.2)

    X_train = train_df[FEATURES]
    y_train = train_df["win"].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df["win"].astype(int)

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver="lbfgs",
            max_iter=8000,
            C=0.7,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(X_train, y_train)

    # explicit metrics
    yhat_train = pipeline.predict(X_train)
    yhat_test = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, yhat_train)
    test_acc = accuracy_score(y_test, yhat_test)

    print(f"[{league}] Rows: train={len(train_df)} test={len(test_df)}")
    print(f"[{league}] Train acc: {train_acc:.3f}  Test acc: {test_acc:.3f}")
    joblib.dump(pipeline, out_path)
    print(f"[{league}] Saved: {out_path}")


def main():
    path = "team_trends_last730d.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    df = pd.read_csv(path)
    train_one(df, "NBA", "moneyline_model_NBA.joblib")
    train_one(df, "NFL", "moneyline_model_NFL.joblib")


if __name__ == "__main__":
    main()
