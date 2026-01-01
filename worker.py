#!/usr/bin/env python3
import os
import time
import logging

from app import app, db
from models import settle_parlays_once  # you should already have this function

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("parlay-worker")

SLEEP_SECONDS = int(os.environ.get("SETTLE_INTERVAL_SECONDS", "300"))  # default 5 min

def main():
    log.info("Parlay worker starting. interval=%ss", SLEEP_SECONDS)

    while True:
        try:
            with app.app_context():
                settled = settle_parlays_once()
                db.session.commit()
                log.info("settle_parlays_once() done. settled=%s", settled)
        except Exception as e:
            log.exception("Worker error: %s", e)
            try:
                db.session.rollback()
            except Exception:
                pass

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
