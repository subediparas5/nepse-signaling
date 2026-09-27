"""
Pull NOTS daily history for every listed equity plus the NEPSE index into data/.

NOTS only serves about one trading year, so run this regularly (the daily workflow does):
anything older survives only in the committed CSVs. Safe to re-run — writes are upserts.

    uv run src/backfill_history.py              # full available window
    uv run src/backfill_history.py --days 30    # recent window only (faster diff, same request count)
    uv run src/backfill_history.py --symbols NABIL NICA
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import history_store
from nepse_official import (
    get_business_date,
    get_index_history,
    get_price_adjustments,
    get_official_listed_stocks,
    get_security_history,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flush to disk every N symbols so a crash mid-run keeps what was fetched.
_FLUSH_EVERY = 50


def backfill(days: int, symbols: set[str] | None = None) -> None:
    business_date = get_business_date()
    end = business_date
    start = (date.fromisoformat(end) - timedelta(days=days)).isoformat()
    logger.info("Backfill %s → %s (business date %s)", start, end, business_date)

    index_rows = [r for r in get_index_history() if start <= (r["date"] or "") <= end]
    logger.info("NEPSE index: %s rows, %s changed", len(index_rows), history_store.upsert_index(index_rows))

    actions = get_price_adjustments()
    changed_actions = history_store.upsert_corporate_actions(actions)
    logger.info("Corporate actions: %s notices, %s changed", len(actions), changed_actions)

    listed = get_official_listed_stocks()
    if symbols:
        listed = [s for s in listed if s["symbol"] in symbols]
    history_store.upsert_securities(listed, seen_on=business_date)

    pending: list[dict] = []
    changed = failed = 0
    for i, sec in enumerate(listed):
        sym, sid = sec["symbol"], sec.get("security_id")
        if sid is None:
            continue
        try:
            rows = get_security_history(int(sid), start, end)
        except Exception as e:
            failed += 1
            logger.warning("History failed %s (id=%s): %s", sym, sid, e)
            time.sleep(0.5)
            continue
        pending.extend({**r, "symbol": sym} for r in rows)
        if (i + 1) % _FLUSH_EVERY == 0:
            changed += history_store.upsert_prices(pending)
            pending = []
            logger.info("Backfill: %s / %s symbols", i + 1, len(listed))
        time.sleep(0.08)
    changed += history_store.upsert_prices(pending)

    logger.info(
        "Backfill done: %s symbols, %s price rows added/changed, %s failed",
        len(listed),
        changed,
        failed,
    )
    if failed and failed > len(listed) // 10:
        raise SystemExit(f"{failed} of {len(listed)} symbols failed — NOTS likely rate-limiting")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=400, help="calendar days back from the business date")
    ap.add_argument("--symbols", nargs="*", help="limit to these symbols")
    args = ap.parse_args()
    backfill(args.days, set(args.symbols) if args.symbols else None)


if __name__ == "__main__":
    main()
