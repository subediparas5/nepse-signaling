"""
Git-friendly on-disk history: plain CSV, deterministic ordering and number formatting.

Layout (under DATA_DIR, default `<repo>/data`):
  prices/YYYY-MM.csv   one row per (date, symbol) — daily OHLCV
  index/nepse.csv      one row per date — NEPSE index
  signals/YYYY-MM.csv  one row per (date, symbol) — rule verdicts from the daily run
  securities.csv       one row per symbol — id, sector, first/last seen
  corporate_actions.csv one row per (date, symbol) — bonus/rights/dividend price adjustments

Prices are stored raw (as traded). features.build_panel applies corporate_actions on load.

Writes are upserts: a row keyed like an existing one is merged field by field, and an
empty incoming value never erases a stored one. Re-running with the same data leaves
files byte-identical, so the scheduled job only produces a diff when something changed.
"""

from __future__ import annotations

import csv
import math
import numbers
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(os.getenv("NEPSE_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")

PRICE_FIELDS = [
    "date", "symbol", "open", "high", "low", "close", "prev_close", "vwap",
    "volume", "turnover", "trades", "week_52_high", "week_52_low",
]
INDEX_FIELDS = [
    "date", "open", "high", "low", "close", "pct_change",
    "turnover", "volume", "trades", "week_52_high", "week_52_low",
]
SIGNAL_FIELDS = [
    "date", "symbol", "sector", "classification", "score", "opportunity_score", "risk_score", "regime",
    # Legacy vote engine, kept so its open/VWAP votes can be evaluated once snapshots accumulate.
    "verdict", "buy_score", "sell_score", "confidence",
    "technical_buy", "technical_sell", "fundamental_buy", "fundamental_sell", "close", "reasons",
]
SECURITY_FIELDS = ["symbol", "security_id", "sector", "first_seen", "last_seen"]
ACTION_FIELDS = ["date", "symbol", "prev_close", "adjusted_price", "factor", "reason", "alert_id"]


def fmt(value: Any) -> str:
    """Stable text for CSV: '' for missing, integers without '.0', floats rounded to 4 dp."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):  # includes numpy floats, whose repr is "np.float64(...)"
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return ""
        r = round(value, 4)
        if r == int(r):
            return str(int(r))
        return repr(r)
    return str(value).strip()


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    tmp.replace(path)


def _merge(old: dict[str, str] | None, new: dict[str, Any], fields: list[str]) -> dict[str, str]:
    out = dict(old or {k: "" for k in fields})
    for k in fields:
        v = fmt(new.get(k))
        if v != "":
            out[k] = v
    return out


def _upsert(path: Path, fields: list[str], key: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> int:
    """Merge rows into one CSV. Returns how many keyed rows were added or changed."""
    existing = {tuple(r[k] for k in key): r for r in _read(path)}
    changed = 0
    for row in rows:
        k = tuple(fmt(row.get(c)) for c in key)
        if any(part == "" for part in k):
            continue
        merged = _merge(existing.get(k), row, fields)
        if merged != existing.get(k):
            existing[k] = merged
            changed += 1
    if changed:
        _write(path, fields, [existing[k] for k in sorted(existing)])
    return changed


def _by_month(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d = fmt(r.get("date"))
        if len(d) >= 7:
            groups[d[:7]].append(r)
    return groups


def upsert_prices(rows: Iterable[dict[str, Any]], data_dir: Path = DATA_DIR) -> int:
    return sum(
        _upsert(data_dir / "prices" / f"{month}.csv", PRICE_FIELDS, ("date", "symbol"), group)
        for month, group in sorted(_by_month(rows).items())
    )


def upsert_signals(rows: Iterable[dict[str, Any]], data_dir: Path = DATA_DIR) -> int:
    return sum(
        _upsert(data_dir / "signals" / f"{month}.csv", SIGNAL_FIELDS, ("date", "symbol"), group)
        for month, group in sorted(_by_month(rows).items())
    )


def upsert_index(rows: Iterable[dict[str, Any]], data_dir: Path = DATA_DIR) -> int:
    return _upsert(data_dir / "index" / "nepse.csv", INDEX_FIELDS, ("date",), rows)


def upsert_securities(rows: Iterable[dict[str, Any]], seen_on: str, data_dir: Path = DATA_DIR) -> int:
    """Record listed securities. Symbols are never removed, so delisted names stay for backtests."""
    path = data_dir / "securities.csv"
    existing = {r["symbol"]: r for r in _read(path)}
    incoming = []
    for r in rows:
        sym = fmt(r.get("symbol"))
        if not sym:
            continue
        prev = existing.get(sym)
        first = prev["first_seen"] if prev and prev.get("first_seen") else seen_on
        first = min(first, seen_on)
        last = max(prev["last_seen"], seen_on) if prev and prev.get("last_seen") else seen_on
        incoming.append({**r, "symbol": sym, "first_seen": first, "last_seen": last})
    return _upsert(path, SECURITY_FIELDS, ("symbol",), incoming)


def upsert_corporate_actions(rows: Iterable[dict[str, Any]], data_dir: Path = DATA_DIR) -> int:
    return _upsert(data_dir / "corporate_actions.csv", ACTION_FIELDS, ("date", "symbol"), rows)


def load_corporate_actions(data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    return _read(data_dir / "corporate_actions.csv")


def load_prices(data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    """All stored price rows, sorted by (date, symbol)."""
    rows: list[dict[str, str]] = []
    for path in sorted((data_dir / "prices").glob("*.csv")):
        rows.extend(_read(path))
    return rows


def load_index(data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    return _read(data_dir / "index" / "nepse.csv")


def stored_dates(data_dir: Path = DATA_DIR) -> set[str]:
    return {r["date"] for r in load_prices(data_dir)}
