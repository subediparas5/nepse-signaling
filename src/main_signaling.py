import json
import os
import re
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from openai import OpenAI

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nepse_signal_rules import TRADABLE_SECTORS, classify_nepse_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPEN_AI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY environment variable is not set.")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NEPSE_TRADING_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Origin": "https://www.nepsetrading.com",
    "Referer": "https://www.nepsetrading.com/",
}

NEPSE_TRADING_SESSION = requests.Session()
NEPSE_TRADING_SESSION.headers.update(NEPSE_TRADING_HEADERS)

STOCKS_LISTED_URL = "https://api.nepsetrading.com/stocks-listed"
FUNDAMENTALS_URL = "https://api.nepsetrading.com/recent-report?"

LLM_FIELDS = [
    "symbol", "sector", "market_cap_category", "ltp",
    "signal_verdict", "signal_buy_score", "signal_sell_score",
    "signal_confidence", "signal_reasons",
    "pe", "pb", "roe", "npl", "eps_ttm",
    "promoter_percentage", "dividend_yield",
    "week_52_high", "week_52_low",
    "open", "high", "low", "vwap", "prev_close",
    "volume", "turnover", "diff_pct", "ma120", "ma180",
]

SYSTEM_PROMPT = """\
You are an expert Nepal stock market analyst. You receive the top BUY candidates \
from NEPSE that passed a strict rule-based screen.

For each stock you get: buy/sell scores, reasons, P/E, P/B, ROE, NPL, EPS, \
promoter %, dividend yield, 52-week range, moving averages, volume, turnover, \
VWAP, and market cap category (Large/Mid/Small).

Your job:
1. Review each candidate and either CONFIRM or REJECT.
2. REJECT if: negative EPS, NPL > 5%, near 52-week high, low promoter (<25%), \
price well above 120d/180d MA (overextended), or very low turnover (illiquid).
3. PREFER stocks with: good dividend yield (>2%), high liquidity, MAs converging \
bullishly, and strong fundamentals.
4. For confirmed picks, write a short reason a retail trader can act on.
5. If promoter > 50%, add "check lock-in expiry" to the reason.
6. Factor in market cap: small caps carry higher risk, note if relevant.
7. Return at most 5 confirmed picks — quality over quantity.

Output format — return ONLY this, nothing else:

SYMBOL | Rs PRICE | reason (15 words max)

One stock per line. No numbering, no markdown, no headers. \
If no stock passes your review, return exactly: "No strong picks today."\
"""


def get_listed_stocks() -> list[dict]:
    """Stock listing with sector codes + promoter % from nepsetrading.com."""
    for attempt in range(3):
        try:
            resp = NEPSE_TRADING_SESSION.get(STOCKS_LISTED_URL)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - stocks-listed: {e}")
    return []


def get_fundamental_lookup() -> dict[str, dict]:
    """Fundamentals (PE, PB, ROE, NPL, EPS) from nepsetrading.com."""
    for attempt in range(3):
        try:
            resp = NEPSE_TRADING_SESSION.get(FUNDAMENTALS_URL)
            resp.raise_for_status()
            rows = resp.json()
            return {
                row["symbol"].strip(): row
                for row in rows
                if row.get("symbol")
            }
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - fundamentals: {e}")
    return {}


_SS_COL_MAP = {
    3: "open", 4: "high", 5: "low", 6: "close", 7: "ltp",
    10: "vwap", 11: "volume", 12: "prev_close",
    13: "turnover", 14: "transactions",
    16: "range", 17: "diff_pct", 18: "range_pct",
    20: "ma120", 21: "ma180",
    22: "week_52_high", 23: "week_52_low",
}


def get_sharesansar_data() -> dict[str, dict]:
    """OHLCV + 52-week range + MAs from ShareSansar (single bulk request)."""
    url = "https://www.sharesansar.com/today-share-price"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        table_m = re.search(
            r'<table[^>]*id="headFixed"[^>]*>(.*?)</table>', resp.text, re.DOTALL
        )
        if not table_m:
            logger.warning("ShareSansar: table not found")
            return {}
        lookup: dict[str, dict] = {}
        for row in re.findall(r'<tr[^>]*>(.*?)</tr>', table_m.group(1), re.DOTALL):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) < 24:
                continue
            clean = [re.sub(r'<[^>]+>', '', c).strip().replace(",", "") for c in cells]
            symbol = clean[1]
            if not symbol or not symbol[0].isalpha():
                continue
            entry: dict = {}
            for idx, key in _SS_COL_MAP.items():
                try:
                    entry[key] = float(clean[idx])
                except (ValueError, IndexError):
                    pass
            lookup[symbol] = entry
        logger.info(f"ShareSansar: {len(lookup)} stocks")
        return lookup
    except Exception as e:
        logger.warning(f"ShareSansar failed: {e}")
        return {}


def _compute_sector_medians(stocks: list[dict]) -> dict[str, float]:
    """Compute median diff_pct per sector for relative-strength scoring."""
    from statistics import median
    sector_vals: dict[str, list[float]] = {}
    for s in stocks:
        dp = s.get("diff_pct")
        sec = s.get("sector")
        if dp is not None and sec:
            try:
                sector_vals.setdefault(sec, []).append(float(dp))
            except (ValueError, TypeError):
                pass
    return {sec: median(vals) for sec, vals in sector_vals.items() if vals}


def get_classified_stocks() -> tuple[list[dict], list[dict], int]:
    """Returns (top_buys, top_sells, total_screened)."""
    listed_stocks = get_listed_stocks()
    fund_lookup = get_fundamental_lookup()
    ss_data = get_sharesansar_data()

    logger.info(
        f"Data sources: {len(listed_stocks)} listed, "
        f"{len(fund_lookup)} fundamentals, {len(ss_data)} ShareSansar"
    )

    pre_stocks: list[dict] = []
    for stock in listed_stocks:
        sector = stock.get("sector")
        symbol = stock.get("symbol")
        if not sector or not symbol:
            continue
        if sector not in TRADABLE_SECTORS:
            continue

        ss = ss_data.get(symbol, {})
        fund = fund_lookup.get(symbol, {})

        data: dict = {**fund, **ss}
        data["sector"] = sector
        data["symbol"] = symbol
        data["promoter_percentage"] = stock.get("promoter_percentage")
        data["public_percentage"] = stock.get("public_percentage")
        data.setdefault("ltp", stock.get("latesttransactionprice"))

        mcap = stock.get("market_capitalization")
        if mcap:
            data["market_capitalization"] = mcap
            try:
                mc = float(mcap)
                if mc >= 20_000_000_000:
                    data["market_cap_category"] = "Large"
                elif mc >= 5_000_000_000:
                    data["market_cap_category"] = "Mid"
                else:
                    data["market_cap_category"] = "Small"
            except (ValueError, TypeError):
                pass

        dpps = fund.get("dpps")
        if dpps is not None:
            data["dpps"] = dpps
            try:
                ltp_val = float(data.get("ltp", 0))
                dpps_val = float(dpps)
                if ltp_val > 0:
                    data["dividend_yield"] = round(dpps_val / ltp_val * 100, 2)
            except (ValueError, TypeError):
                pass

        pre_stocks.append(data)

    sector_medians = _compute_sector_medians(pre_stocks)

    all_stocks: list[dict] = []
    for data in pre_stocks:
        data["_sector_median_diff"] = sector_medians.get(data["sector"])
        data.update(classify_nepse_signal(data, data["sector"]))
        all_stocks.append(data)

    buys = [s for s in all_stocks if s.get("signal_verdict") == "BUY"]
    buys.sort(
        key=lambda s: s.get("signal_buy_score", 0) - s.get("signal_sell_score", 0),
        reverse=True,
    )
    buys = buys[:10]

    sells = [s for s in all_stocks if s.get("signal_verdict") == "SELL"]
    sells.sort(
        key=lambda s: s.get("signal_sell_score", 0) - s.get("signal_buy_score", 0),
        reverse=True,
    )
    sells = sells[:5]

    logger.info(
        f"Classified {len(all_stocks)} stocks -> "
        f"{len(buys)} top BUY, {len(sells)} top SELL"
    )
    return buys, sells, len(all_stocks)


def compact_for_llm(candidates: list[dict]) -> list[dict]:
    return [
        {k: s.get(k) for k in LLM_FIELDS if s.get(k) is not None}
        for s in candidates
    ]


def get_llm_picks(candidates: list[dict]) -> str:
    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")
    payload = compact_for_llm(candidates)
    user_content = json.dumps(payload, default=str)
    logger.info(f"Sending {len(payload)} BUY candidates to DeepSeek")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


def _npt_now() -> str:
    npt = timezone(timedelta(hours=5, minutes=45))
    return datetime.now(npt).strftime("%Y-%m-%d %H:%M NPT")


def format_top_picks_message(llm_output: str, n_screened: int) -> str:
    lines = [
        f"🔥 *Top NEPSE Picks — {_npt_now()}*",
        f"_{n_screened} stocks screened_",
        "",
    ]

    picks = [l.strip() for l in llm_output.strip().splitlines() if l.strip()]
    if picks and picks[0].lower().startswith("no strong"):
        lines.append("No strong picks today.")
    else:
        for p in picks:
            parts = [x.strip() for x in p.split("|", 2)]
            if len(parts) == 3:
                sym, price, reason = parts
                lines.append(f"✅ *{sym}* — {price}")
                lines.append(f"    _{reason}_")
                lines.append("")
            else:
                lines.append(p)

    lines.append("_Not financial advice. DYOR._")
    return "\n".join(lines)


def _fmt_num(val, decimals=1) -> str:
    """Safely format a number, handling raw API floats."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _format_stock_block(s: dict) -> list[str]:
    sym = s.get("symbol", "?")
    price = _fmt_num(s.get("ltp"), 1)
    pe = s.get("pe")
    roe = s.get("roe")
    eps = s.get("eps_ttm")
    promo = s.get("promoter_percentage")
    dy = s.get("dividend_yield")
    mcap = s.get("market_cap_category")
    buy_sc = s.get("signal_buy_score", 0)
    sell_sc = s.get("signal_sell_score", 0)
    conf = s.get("signal_confidence", 0)
    reasons = s.get("signal_reasons", "")

    stats = []
    if pe is not None:
        stats.append(f"PE {_fmt_num(pe)}")
    if roe is not None:
        stats.append(f"ROE {_fmt_num(roe)}%")
    if eps is not None:
        stats.append(f"EPS {_fmt_num(eps, 2)}")
    if promo is not None:
        stats.append(f"Promo {_fmt_num(promo, 0)}%")
    if dy is not None and dy > 0:
        stats.append(f"DY {dy:.1f}%")
    stats_str = " | ".join(stats) if stats else "—"

    cap_tag = f" [{mcap}]" if mcap else ""
    lines = [
        f"*{sym}*{cap_tag} — Rs {price}  (+{buy_sc}/−{sell_sc}, {conf}% conf)",
        f"    {stats_str}",
    ]
    if reasons:
        top_reasons = reasons.split(" | ")[:5]
        lines.append(f"    _{', '.join(top_reasons)}_")
    lines.append("")
    return lines


def format_all_buys_message(buys: list[dict]) -> str:
    lines = [f"📊 *All BUY Signals — {_npt_now()}*", ""]
    for s in buys:
        lines.extend(_format_stock_block(s))
    return "\n".join(lines)


def format_sell_watchlist_message(sells: list[dict]) -> str:
    if not sells:
        return ""
    lines = [f"⚠️ *SELL Watchlist — {_npt_now()}*", ""]
    for s in sells:
        lines.extend(_format_stock_block(s))
    lines.append("_Stocks to avoid or consider exiting._")
    return "\n".join(lines)


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4096
    chunks = [message[i:i + max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown",
        })
        if not resp.ok:
            logger.error(f"Telegram send failed: {resp.status_code} {resp.text}")
        else:
            logger.info("Telegram message sent")


if __name__ == "__main__":
    buys, sells, n_screened = get_classified_stocks()

    if not buys:
        send_telegram("No strong BUY signals found today.")
    else:
        llm_output = get_llm_picks(buys)
        logger.info(f"LLM output:\n{llm_output}")

        send_telegram(format_top_picks_message(llm_output, n_screened))
        send_telegram(format_all_buys_message(buys))

    sell_msg = format_sell_watchlist_message(sells)
    if sell_msg:
        send_telegram(sell_msg)
