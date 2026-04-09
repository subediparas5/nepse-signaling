import json
import os
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en",
    "Accept-Encoding": "gzip, deflate",
    "Origin": "https://www.nepsetrading.com",
    "Connection": "keep-alive",
    "Referer": "https://www.nepsetrading.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Priority": "u=0",
    "TE": "trailers",
}

NEPSE_TRADING_SESSION = requests.Session()
NEPSE_TRADING_SESSION.headers.update(HEADERS)

STOCK_DETAILS_URL = "https://api.nepsetrading.com/sidebar?code={stock_symbol}"
SECTOR_DETAILS_BASE_URL = "https://api.nepsetrading.com/stocks-listed"
FUNDAMENTAL_DETAILS_URL = "https://api.nepsetrading.com/recent-report?"

UNNECESSARY_SIDEBAR_KEYS = frozenset({
    "bb_signal", "fib_signal", "fib_range",
    "ema_signal", "sar_signal",
    "trade_signal", "trend_confirmation", "obv_breakout",
    "daily_volatility_rs", "weekly_volatility_rs", "monthly_volatility_rs",
    "avg_volume_3_days",
    "eps", "dpps", "ltp",
})

LLM_FIELDS = [
    "symbol", "sector", "latesttransactionprice",
    "signal_verdict", "signal_buy_score", "signal_sell_score", "signal_reasons",
    "pe", "pb", "roe", "npl", "eps_ttm",
    "promoter_percentage", "share_float",
    "week_52_high", "week_52_low",
    "one_month_perf", "three_month_perf",
    "technical_rating", "supertrend_signal",
    "market_sentiment", "market_trend",
    "obv_price_divergence", "divident_yeild",
    "beta_yearly",
    "volume", "avg_volume_30_days",
]

SYSTEM_PROMPT = """\
You are an expert Nepal stock market analyst. You receive the top BUY candidates \
from NEPSE that passed a strict rule-based screen.

For each stock you get: buy/sell scores, reasons, P/E, P/B, ROE, NPL, EPS, \
promoter %, 52-week range, momentum, volume, market sentiment, etc.

Your job:
1. Review each candidate and either CONFIRM or REJECT.
2. REJECT if: negative EPS, NPL > 5%, near 52-week high, low promoter (<25%), \
or weak/negative momentum.
3. For confirmed picks, write a short reason a retail trader can act on.
4. If promoter > 50%, add "check lock-in expiry" to the reason.
5. Return at most 5 confirmed picks — quality over quantity.

Output format — return ONLY this, nothing else:

SYMBOL | Rs PRICE | reason (15 words max)

One stock per line. No numbering, no markdown, no headers. \
If no stock passes your review, return exactly: "No strong picks today."\
"""


def get_listed_stocks_with_sector() -> list:
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(SECTOR_DETAILS_BASE_URL)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Error fetching stock info: {e}")
    return []


def get_fundamental_lookup() -> dict[str, dict]:
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(FUNDAMENTAL_DETAILS_URL)
            response.raise_for_status()
            rows = response.json()
            return {
                row["symbol"].strip(): row
                for row in rows
                if row.get("symbol")
            }
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Error fetching fundamental data: {e}")
    return {}


def get_stock_info(stock_symbol: str) -> dict:
    for attempt in range(3):
        try:
            response = NEPSE_TRADING_SESSION.get(STOCK_DETAILS_URL.format(stock_symbol=stock_symbol))
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            raw = response.json()
            cleaned: dict = {}
            for key, value in raw.items():
                if key in UNNECESSARY_SIDEBAR_KEYS:
                    continue
                if isinstance(value, float):
                    cleaned[key] = round(value, 2)
                elif isinstance(value, str):
                    try:
                        cleaned[key] = round(float(value), 2)
                    except ValueError:
                        cleaned[key] = value
                else:
                    cleaned[key] = value
            return cleaned
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Skipping {stock_symbol}: {e}")
                return {}
    return {}


def get_classified_stocks() -> tuple[list[dict], int]:
    listed_stocks = get_listed_stocks_with_sector()
    fund_lookup = get_fundamental_lookup()
    all_stocks: list[dict] = []

    for stock in listed_stocks:
        sector = stock.get("sector")
        symbol = stock.get("symbol")
        if not sector or not symbol:
            continue
        if sector not in TRADABLE_SECTORS:
            continue

        data: dict = get_stock_info(symbol)

        fund_row = fund_lookup.get(symbol, {})
        for k, v in fund_row.items():
            if k not in data:
                data[k] = v

        if not data:
            continue

        data["sector"] = sector
        data["symbol"] = data.get("symbol", symbol)
        data["promoter_percentage"] = stock.get("promoter_percentage")
        data["public_percentage"] = stock.get("public_percentage")
        data["latesttransactionprice"] = data.get(
            "latesttransactionprice", stock.get("latesttransactionprice")
        )

        data.update(classify_nepse_signal(data, sector))
        all_stocks.append(data)

    buys = [
        s for s in all_stocks
        if s.get("signal_verdict") == "BUY"
    ]
    buys.sort(key=lambda s: s.get("signal_buy_score", 0) - s.get("signal_sell_score", 0), reverse=True)
    buys = buys[:10]
    logger.info(f"Classified {len(all_stocks)} stocks -> {len(buys)} top BUY candidates")
    return buys, len(all_stocks)


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


def format_all_buys_message(buys: list[dict]) -> str:
    lines = [
        f"📊 *All BUY Signals — {_npt_now()}*",
        "",
    ]
    for s in buys:
        sym = s.get("symbol", "?")
        price = s.get("latesttransactionprice", "—")
        pe = s.get("pe")
        roe = s.get("roe")
        eps = s.get("eps_ttm")
        promo = s.get("promoter_percentage")
        buy_sc = s.get("signal_buy_score", 0)
        sell_sc = s.get("signal_sell_score", 0)
        reasons = s.get("signal_reasons", [])

        stats = []
        if pe is not None:
            stats.append(f"PE {pe}")
        if roe is not None:
            stats.append(f"ROE {roe}%")
        if eps is not None:
            stats.append(f"EPS {eps}")
        if promo is not None:
            stats.append(f"Promo {promo}%")
        stats_str = " | ".join(stats) if stats else "—"

        lines.append(f"*{sym}* — Rs {price}  (Score +{buy_sc}/−{sell_sc})")
        lines.append(f"    {stats_str}")
        if reasons:
            lines.append(f"    _{', '.join(reasons[:4])}_")
        lines.append("")

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
    buys, n_screened = get_classified_stocks()

    if not buys:
        send_telegram("No strong BUY signals found today.")
    else:
        llm_output = get_llm_picks(buys)
        logger.info(f"LLM output:\n{llm_output}")

        send_telegram(format_top_picks_message(llm_output, n_screened))
        send_telegram(format_all_buys_message(buys))
