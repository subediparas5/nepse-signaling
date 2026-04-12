import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

OFFICIAL_SECTOR_TO_CODE: dict[str, str] = {
    "Commercial Banks": "BANKING",
    "Development Bank Limited": "DEVBANK",
    "Finance": "FINANCE",
    "Microfinance": "MICROFINANCE",
    "Hydro Power": "HYDROPOWER",
    "Manufacturing And Processing": "MANUFACTURE",
    "Hotels And Tourism": "HOTELS",
    "Tradings": "TRADING",
    "Life Insurance": "LIFEINSU",
    "Non-Life Insurance": "NONLIFEINSU",
    "Investment": "INVESTMENT",
    "Others": "OTHERS",
}

_nepse: Any = None


def _client():
    global _nepse
    if _nepse is None:
        from nepse_data_api import Nepse

        _nepse = Nepse(enable_cache=True, cache_ttl=300)
    return _nepse


def get_official_listed_stocks() -> list[dict]:
    """Active equities with sector mapped to `TRADABLE_SECTORS` codes."""
    n = _client()
    out: list[dict] = []
    for row in n.get_company_list():
        if row.get("status") != "A":
            continue
        if row.get("instrumentType") != "Equity":
            continue
        sym = row.get("symbol")
        if not sym:
            continue
        sector = OFFICIAL_SECTOR_TO_CODE.get(row.get("sectorName") or "")
        if not sector:
            continue
        out.append(
            {
                "symbol": sym,
                "sector": sector,
                "promoter_percentage": None,
                "public_percentage": None,
                "market_capitalization": None,
                "latesttransactionprice": None,
            }
        )
    logger.info("NEPSE official listing: %s equities", len(out))
    return out


def _float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_official_share_price_lookup(symbols: set[str] | None = None) -> dict[str, dict]:
    """
    Per-symbol OHLCV-style fields for `main_signaling` / `nepse_signal_rules`.

    Merges `get_stocks()` with `get_security_details()` for 52-week range and
    trade counts.
    """
    n = _client()
    live_rows = n.get_stocks() or []
    live_by_symbol = {r["symbol"]: r for r in live_rows if r.get("symbol")}

    companies = [
        r
        for r in n.get_company_list()
        if r.get("status") == "A"
        and r.get("instrumentType") == "Equity"
        and OFFICIAL_SECTOR_TO_CODE.get(r.get("sectorName") or "")
    ]
    if symbols is not None:
        companies = [r for r in companies if r.get("symbol") in symbols]

    out: dict[str, dict] = {}
    for i, row in enumerate(companies):
        sym = row.get("symbol")
        sid = row.get("id")
        if not sym or sid is None:
            continue
        if i and i % 50 == 0:
            logger.info("NEPSE official market: %s / %s", i, len(companies))
        try:
            detail = n.get_security_details(int(sid), use_cache=True)
        except Exception as e:
            logger.warning("NEPSE security detail failed %s: %s", sym, e)
            time.sleep(0.1)
            continue
        mcs = detail.get("securityMcsData") or {}
        lv = live_by_symbol.get(sym, {})

        op = _float(mcs.get("openPrice") or lv.get("openPrice"))
        hi = _float(mcs.get("highPrice") or lv.get("highPrice"))
        lo = _float(mcs.get("lowPrice") or lv.get("lowPrice"))
        cls = _float(mcs.get("closePrice"))
        ltp = _float(mcs.get("lastTradedPrice") or lv.get("lastTradedPrice"))
        prev = _float(mcs.get("previousClose") or lv.get("previousClose"))
        vol = _float(mcs.get("totalTradeQuantity") or lv.get("totalTradeQuantity"))
        turnover = _float(lv.get("totalTradeValue"))
        vwap = _float(lv.get("averageTradedPrice"))
        diff_pct = _float(lv.get("percentageChange"))
        tx = _float(mcs.get("totalTrades"))

        w52h = _float(mcs.get("fiftyTwoWeekHigh"))
        w52l = _float(mcs.get("fiftyTwoWeekLow"))

        range_abs = None
        range_pct = None
        if hi is not None and lo is not None:
            range_abs = hi - lo
            if prev and prev > 0:
                range_pct = range_abs / prev * 100.0

        entry: dict[str, Any] = {
            "open": op,
            "high": hi,
            "low": lo,
            "close": cls if cls is not None else ltp,
            "ltp": ltp,
            "vwap": vwap,
            "volume": vol,
            "prev_close": prev,
            "turnover": turnover,
            "transactions": tx,
            "diff_pct": diff_pct,
            "range": range_abs,
            "range_pct": range_pct,
            "week_52_high": w52h,
            "week_52_low": w52l,
        }
        out[sym] = {k: v for k, v in entry.items() if v is not None}
        time.sleep(0.03)

    logger.info("NEPSE official market: %s symbols with detail", len(out))
    return out
