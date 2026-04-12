import contextlib
import io
import logging
import time
import types
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def _get_security_details_quiet(n: Any, security_id: int, *, use_cache: bool) -> dict:
    """`nepse-data-api` prints errors to stdout; swallow so logs stay readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return n.get_security_details(security_id, use_cache=use_cache)


def _refresh_auth_token_quiet(n: Any) -> Any:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return n.refresh_auth_token()


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

# Refresh Salter during long detail loops so tokens do not expire mid-batch.
_PROACTIVE_REFRESH_EVERY = 32


def _inject_session_timeouts(session: Any, connect: float = 10.0, read: float = 25.0) -> None:
    """Nepse calls `session.get` without `timeout` — a stalled TLS read blocks forever."""
    if getattr(session, "_nepse_signaling_timeout_injected", False):
        return
    t = (connect, read)

    def request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = t
        return requests.Session.request(self, method, url, **kwargs)

    session.request = types.MethodType(request, session)
    setattr(session, "_nepse_signaling_timeout_injected", True)


def _configure_nepse_session(session: Any) -> None:
    """NOTS often drops connections under burst traffic; urllib3 retries help GET/POST."""
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=0.65,
        status_forcelist=(429, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)


def _client():
    global _nepse
    if _nepse is None:
        from nepse_data_api import Nepse

        _nepse = Nepse(enable_cache=True, cache_ttl=300)
        _configure_nepse_session(_nepse.session)
        _inject_session_timeouts(_nepse.session)
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


def _detail_fetch_failed(detail: Any) -> bool:
    """`nepse_data_api` returns {} on 401 and on request errors; treat as retry."""
    return not isinstance(detail, dict) or len(detail) == 0


def _authenticate_safe(n: Any, label: str) -> None:
    last: Exception | None = None
    for attempt in range(5):
        try:
            n.authenticate()
            return
        except Exception as e:
            last = e
            wait = 1.0 * (attempt + 1)
            logger.warning("NEPSE authenticate failed (%s), retry in %.1fs: %s", label, wait, e)
            time.sleep(wait)
    if last:
        raise last


# After re-auth, a few tries catch blips; empty JSON usually needs time/new session, not 8+ GETs.
_POST_AUTH_DETAIL_ROUNDS = 2


def _poll_security_detail(
    n: Any, security_id: int, *, use_cache_first: bool, rounds: int = _POST_AUTH_DETAIL_ROUNDS
) -> dict:
    """Short retry after a fresh token — repeated empty bodies are NOTS/WAF, not fixed by spamming."""
    use_cache = use_cache_first
    d: dict = {}
    for r in range(rounds):
        d = _get_security_details_quiet(n, security_id, use_cache=use_cache)
        use_cache = False
        if not _detail_fetch_failed(d):
            return d
        if r < rounds - 1:
            time.sleep(0.55 + 0.5 * r)
    return d


def _detail_pre_auth_attempts(n: Any, security_id: int) -> dict:
    """At most two GETs with current token (401 will repeat identically — do not poll 4×)."""
    d = _get_security_details_quiet(n, security_id, use_cache=True)
    if not _detail_fetch_failed(d):
        return d
    time.sleep(0.22)
    return _get_security_details_quiet(n, security_id, use_cache=False)


def _proactive_refresh_token(n: Any, index: int, batch: int) -> None:
    try:
        if not getattr(n, "refresh_token", None):
            return
        td = _refresh_auth_token_quiet(n)
        if isinstance(td, dict) and td.get("accessToken"):
            logger.info(
                "NEPSE proactive refresh-token (%s / %s symbols)",
                index,
                batch,
            )
    except Exception as e:
        logger.warning("NEPSE proactive refresh at %s failed: %s", index, e)


def _fetch_security_detail(n: Any, security_id: int, symbol: str) -> tuple[dict, Any]:
    """GET security detail; re-authenticate if token expired (401 → {})."""
    d = _detail_pre_auth_attempts(n, security_id)
    if not _detail_fetch_failed(d):
        return d, n
    logger.warning("NEPSE detail failed for %s (id=%s) — refreshing session", symbol, security_id)
    refresh_ok = False
    if getattr(n, "refresh_token", None):
        td = _refresh_auth_token_quiet(n)
        refresh_ok = isinstance(td, dict) and bool(td.get("accessToken"))
        if refresh_ok:
            d = _poll_security_detail(n, security_id, use_cache_first=False)
            if not _detail_fetch_failed(d):
                return d, n
    if refresh_ok:
        time.sleep(0.5)
    _authenticate_safe(n, symbol)
    d = _poll_security_detail(n, security_id, use_cache_first=False)
    if not _detail_fetch_failed(d):
        return d, n
    global _nepse
    _nepse = None
    time.sleep(0.6)
    n2 = _client()
    _authenticate_safe(n2, f"{symbol}-new-client")
    d = _poll_security_detail(n2, security_id, use_cache_first=False)
    return d, n2


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
        if i > 0 and i % _PROACTIVE_REFRESH_EVERY == 0:
            _proactive_refresh_token(n, i, len(companies))
        try:
            detail, n = _fetch_security_detail(n, int(sid), sym)
        except Exception as e:
            logger.warning("NEPSE security detail failed %s: %s", sym, e)
            time.sleep(0.1)
            continue
        if _detail_fetch_failed(detail):
            logger.warning("NEPSE no detail payload for %s after retry — skipping", sym)
            time.sleep(0.05)
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
        time.sleep(0.08)

    logger.info("NEPSE official market: %s symbols with detail", len(out))
    return out
