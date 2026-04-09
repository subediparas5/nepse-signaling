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
    "Accept-Encoding": "gzip, deflate, br, zstd",
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
You are an expert Nepal stock market analyst. You receive two groups of NEPSE stocks:

**STRONG signals** (BUY/SELL) — high-conviction picks from the rule engine.
**LEAN signals** (LEAN_BUY/LEAN_SELL) — weaker/mixed signals worth a second look.

For each stock you get: signal_verdict, buy/sell scores, reasons, P/E, P/B, ROE, \
NPL, EPS, promoter %, 52-week range, momentum, volume, market sentiment, etc.

Your job:
1. For STRONG BUY — confirm or downgrade to HOLD. Red flags: negative EPS, \
high NPL (>5%), weak momentum, low promoter (<25%), near 52w high.
2. For STRONG SELL — confirm or upgrade to HOLD. Look for contrarian value: \
near 52w low + improving EPS + accumulation.
3. For LEAN signals — promote to BUY/SELL only if the data strongly supports it. \
Otherwise mark HOLD.
4. If promoter holding is high (>50%), remind reader to check lock-in expiry.
5. Never recommend a stock with negative EPS as BUY unless there's a clear \
turnaround story in the metrics.

Output format — return ONLY lines in this exact format, nothing else:
SYMBOL (BUY|SELL|HOLD): <concise 10-20 word reason>

Group BUY first, then SELL, then HOLD. No markdown, no headers, no extra text.\
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
    for attempt in range(5):
        try:
            response = NEPSE_TRADING_SESSION.get(STOCK_DETAILS_URL.format(stock_symbol=stock_symbol))
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
            logger.error(f"Attempt {attempt + 1} - Error fetching sidebar for {stock_symbol}: {e}")
            if attempt == 4:
                return {}
    return {}


def get_classified_stocks() -> list[dict]:
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

        sidebar = get_stock_info(symbol)
        if not sidebar:
            continue

        fund_row = fund_lookup.get(symbol, {})
        for k, v in fund_row.items():
            if k not in sidebar:
                sidebar[k] = v

        sidebar["sector"] = sector
        sidebar["promoter_percentage"] = stock.get("promoter_percentage")
        sidebar["public_percentage"] = stock.get("public_percentage")

        sidebar.update(classify_nepse_signal(sidebar, sector))
        all_stocks.append(sidebar)

    strong = [s for s in all_stocks if s.get("signal_verdict") in ("BUY", "SELL")]
    lean = [s for s in all_stocks if s.get("signal_verdict") in ("LEAN_BUY", "LEAN_SELL")]
    lean.sort(
        key=lambda s: abs(s.get("signal_buy_score", 0) - s.get("signal_sell_score", 0)),
        reverse=True,
    )
    lean = lean[:30]
    logger.info(
        f"Classified {len(all_stocks)} stocks -> "
        f"{len(strong)} strong + {len(lean)} lean candidates"
    )
    return strong, lean


def compact_for_llm(candidates: list[dict]) -> list[dict]:
    return [
        {k: s.get(k) for k in LLM_FIELDS if s.get(k) is not None}
        for s in candidates
    ]


def get_llm_reasons(strong: list[dict], lean: list[dict]) -> str:
    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")
    payload = {
        "strong_signals": compact_for_llm(strong),
        "lean_signals": compact_for_llm(lean),
    }
    user_content = json.dumps(payload, default=str)
    total = len(strong) + len(lean)
    logger.info(f"Sending {total} stocks to DeepSeek ({len(strong)} strong + {len(lean)} lean)")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


def format_telegram_message(llm_output: str, n_candidates: int) -> str:
    npt = timezone(timedelta(hours=5, minutes=45))
    now = datetime.now(npt).strftime("%Y-%m-%d %H:%M NPT")
    header = f"*NEPSE Signal Report — {now}*\n_{n_candidates} stocks screened_\n\n"
    return header + llm_output


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
    strong, lean = get_classified_stocks()

    if not strong and not lean:
        msg = "No actionable signals found today."
        logger.info(msg)
        send_telegram(msg)
    else:
        llm_output = get_llm_reasons(strong, lean)
        logger.info(f"LLM output:\n{llm_output}")

        total = len(strong) + len(lean)
        tg_msg = format_telegram_message(llm_output, total)
        send_telegram(tg_msg)
