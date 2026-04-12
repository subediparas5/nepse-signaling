import html
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

from nepse_official import get_official_listed_stocks, get_official_share_price_lookup
from nepse_signal_rules import TRADABLE_SECTORS, classify_nepse_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPEN_AI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY environment variable is not set.")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_DM_CHAT_ID = os.getenv("TELEGRAM_DM_CHAT_ID")
_raw_send = (os.getenv("TELEGRAM_SEND_TO") or "both").strip().lower()
TELEGRAM_SEND_TO = _raw_send if _raw_send in ("group", "dm", "both") else "both"
if _raw_send not in ("group", "dm", "both", ""):
    logger.warning("Unknown TELEGRAM_SEND_TO=%r; using both.", _raw_send)

GROUP_BUY_TOP_N = 5

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
You are an expert Nepal stock market analyst. You receive BUY candidates from NEPSE \
that passed a rule-based screen using official exchange price data (no P/E or EPS in the feed).

For each stock you get: buy/sell scores, reasons, 52-week range, OHLC, VWAP, volume, turnover, \
and sector. Moving averages may be absent.

Your job:
1. Review each candidate and either CONFIRM or REJECT.
2. REJECT if: near 52-week high, very low turnover (illiquid), or price far above VWAP (chase risk).
3. PREFER stocks with: good liquidity, constructive price vs 52-week range, sensible sector context.
4. For confirmed picks, write a short reason a retail trader can act on.
5. Return at most 5 confirmed picks — quality over quantity.

Output format — return ONLY this, nothing else:

SYMBOL | Rs PRICE | reason (15 words max)

One stock per line. No numbering, no markdown, no headers. \
If no stock passes your review, return exactly: "No strong picks today."\
"""


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


def _near_52w_low(stock: dict) -> bool:
    """Within bottom 10% of 52-week range; skips IPO when MA120 is present and zero."""
    try:
        ltp = float(stock.get("ltp", 0))
        hi = float(stock.get("week_52_high", 0))
        lo = float(stock.get("week_52_low", 0))
    except (ValueError, TypeError):
        return False
    if hi <= lo or ltp <= 0:
        return False
    ma120 = stock.get("ma120")
    if ma120 is not None:
        try:
            if float(ma120) == 0:
                return False
        except (ValueError, TypeError):
            pass
    pct_from_low = (ltp - lo) / (hi - lo)
    eps = stock.get("eps_ttm") or stock.get("eps")
    if eps is not None:
        try:
            if float(eps) <= 0:
                return False
        except (ValueError, TypeError):
            return False
    return pct_from_low <= 0.10


def get_classified_stocks() -> tuple[list[dict], list[dict], list[dict], int]:
    """Returns (top_buys, top_sells, near_52w_lows, total_screened).

    Data: Nepal Stock Exchange NOTS only (www.nepalstock.com.np via nepse-data-api).
    """
    listed_stocks = get_official_listed_stocks()
    listed_symbols = {s["symbol"] for s in listed_stocks if s.get("symbol")}

    logger.info(
        "Fetching official NEPSE market detail per symbol (~1–3 min first run)."
    )
    market_by_symbol = get_official_share_price_lookup(
        listed_symbols if listed_symbols else None
    )

    logger.info(
        "Data: NEPSE NOTS — %s listed, %s with market fields",
        len(listed_stocks),
        len(market_by_symbol),
    )

    pre_stocks: list[dict] = []
    for stock in listed_stocks:
        sector = stock.get("sector")
        symbol = stock.get("symbol")
        if not sector or not symbol:
            continue
        if sector not in TRADABLE_SECTORS:
            continue

        mkt = market_by_symbol.get(symbol, {})
        data: dict = {**mkt}
        data["sector"] = sector
        data["symbol"] = symbol
        data["promoter_percentage"] = stock.get("promoter_percentage")
        data["public_percentage"] = stock.get("public_percentage")
        data.setdefault("ltp", stock.get("latesttransactionprice"))

        pre_stocks.append(data)

    sector_medians = _compute_sector_medians(pre_stocks)

    all_stocks: list[dict] = []
    ipo_count = 0
    for data in pre_stocks:
        data["_sector_median_diff"] = sector_medians.get(data["sector"])
        data.update(classify_nepse_signal(data, data["sector"]))
        if data.get("signal_verdict") == "IPO":
            ipo_count += 1
        all_stocks.append(data)

    established = [s for s in all_stocks if s.get("signal_verdict") != "IPO"]

    buys = [s for s in established if s.get("signal_verdict") == "BUY"]
    buys.sort(
        key=lambda s: s.get("signal_buy_score", 0) - s.get("signal_sell_score", 0),
        reverse=True,
    )
    buys = buys[:10]

    sells = [s for s in established if s.get("signal_verdict") == "SELL"]
    sells.sort(
        key=lambda s: s.get("signal_sell_score", 0) - s.get("signal_buy_score", 0),
        reverse=True,
    )
    sells = sells[:5]

    near_lows = [s for s in established if _near_52w_low(s)]
    near_lows.sort(
        key=lambda s: (
            (float(s.get("ltp", 0)) - float(s.get("week_52_low", 0)))
            / (float(s.get("week_52_high", 1)) - float(s.get("week_52_low", 0)))
            if float(s.get("week_52_high", 0)) > float(s.get("week_52_low", 0))
            else 1
        ),
    )

    logger.info(
        "Classified %s stocks (%s IPO excluded) -> %s BUY, %s SELL, %s near 52w low",
        len(all_stocks),
        ipo_count,
        len(buys),
        len(sells),
        len(near_lows),
    )
    return buys, sells, near_lows, len(established)


def compact_for_llm(candidates: list[dict]) -> list[dict]:
    return [
        {k: s.get(k) for k in LLM_FIELDS if s.get(k) is not None}
        for s in candidates
    ]


def get_llm_picks(candidates: list[dict]) -> str:
    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")
    payload = compact_for_llm(candidates)
    user_content = json.dumps(payload, default=str)
    logger.info("Sending %s BUY candidates to DeepSeek", len(payload))

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


def _fmt_num(val, decimals=1) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _pct_from_52w_low(s: dict) -> float:
    try:
        ltp_f = float(s.get("ltp", 0))
        lo_f = float(s.get("week_52_low", 0))
        hi_f = float(s.get("week_52_high", 1))
        if hi_f <= lo_f:
            return 0.0
        return (ltp_f - lo_f) / (hi_f - lo_f) * 100.0
    except (ValueError, TypeError):
        return 0.0


def _mono_block(lines: list[str]) -> str:
    body = "\n".join(html.escape(line, quote=False) for line in lines)
    return f"<pre>{body}</pre>"


def format_group_digest(
    buys: list[dict],
    near_lows: list[dict],
    llm_output: str,
    top_n: int = GROUP_BUY_TOP_N,
) -> str:
    """Short HTML: top BUY rows + 52w-low table for the group chat."""
    ts = html.escape(_npt_now(), quote=False)
    parts = [f"<b>NEPSE</b> · <code>{ts}</code>"]

    if buys:
        rows = [f"{'SYM':<7} {'Rs':>7} {'B/S':>5} {'%':>3}"]
        rows.append("-" * 26)
        for s in buys[:top_n]:
            sym = str(s.get("symbol", "?"))[:7]
            ltp = _fmt_num(s.get("ltp"), 1)
            b = int(s.get("signal_buy_score", 0) or 0)
            sl = int(s.get("signal_sell_score", 0) or 0)
            cf = int(s.get("signal_confidence", 0) or 0)
            bs = f"{b}/{sl}"
            rows.append(f"{sym:<7} {ltp:>7} {bs:>5} {cf:>3}")
        llm_syms: list[str] = []
        for line in llm_output.strip().splitlines():
            line = line.strip()
            if not line or line.lower().startswith("no strong"):
                continue
            seg = [x.strip() for x in line.split("|", 1)]
            if seg and seg[0]:
                llm_syms.append(seg[0].split()[0][:7])
        if llm_syms:
            parts.append("<b>LLM</b> " + " ".join(f"<code>{html.escape(x)}</code>" for x in llm_syms[:5]))
        parts.append("<b>BUY</b> (top {})".format(min(top_n, len(buys))))
        parts.append(_mono_block(rows))
    else:
        parts.append("<b>BUY</b> — none today.")

    if near_lows:
        rows = [f"{'SYM':<7} {'Rs':>7} {'rng%':>4}"]
        rows.append("-" * 22)
        for s in near_lows[:8]:
            sym = str(s.get("symbol", "?"))[:7]
            ltp = _fmt_num(s.get("ltp"), 1)
            p = int(round(_pct_from_52w_low(s)))
            rows.append(f"{sym:<7} {ltp:>7} {p:>4}")
        shown = min(8, len(near_lows))
        parts.append(f"<b>52W LOW</b> ({shown}/{len(near_lows)})")
        parts.append(_mono_block(rows))
    else:
        parts.append("<b>52W LOW</b> — none.")

    parts.append("<i>Not advice.</i>")
    return "\n".join(parts)


def format_personal_report(
    llm_output: str,
    buys: list[dict],
    near_lows: list[dict],
    n_screened: int,
) -> str:
    """Full HTML report for DM: LLM text, all BUY rows, 52w list."""
    ts = html.escape(_npt_now(), quote=False)
    parts = [
        f"<b>NEPSE — full report</b>",
        f"<code>{ts}</code> · screened <b>{n_screened}</b>",
        "",
    ]

    picks = [l.strip() for l in llm_output.strip().splitlines() if l.strip()]
    parts.append("<b>1 · LLM</b>")
    if picks and picks[0].lower().startswith("no strong"):
        parts.append("<i>No picks.</i>")
    else:
        for p in picks[:8]:
            parts.append("· " + html.escape(p, quote=False))
    parts.append("")

    parts.append("<b>2 · BUY candidates</b> ({})".format(len(buys)))
    if buys:
        rows = [
            f"{'SYM':<7} {'Rs':>7} {'B/S':>5} {'cf':>3}  sector",
            "-" * 44,
        ]
        for s in buys:
            sym = str(s.get("symbol", "?"))[:7]
            ltp = _fmt_num(s.get("ltp"), 1)
            b = int(s.get("signal_buy_score", 0) or 0)
            sl = int(s.get("signal_sell_score", 0) or 0)
            cf = int(s.get("signal_confidence", 0) or 0)
            sec = str(s.get("sector", ""))[:12]
            bs = f"{b}/{sl}"
            rows.append(f"{sym:<7} {ltp:>7} {bs:>5} {cf:>3}  {sec}")
        parts.append(_mono_block(rows))
        for s in buys[:12]:
            sym = html.escape(str(s.get("symbol", "?")), quote=False)
            rs = s.get("signal_reasons") or ""
            if rs:
                rshort = html.escape(rs[:220] + ("…" if len(rs) > 220 else ""), quote=False)
                parts.append(f"<b>{sym}</b> <code>{rshort}</code>")
        parts.append("")
    else:
        parts.append("<i>None.</i>\n")

    parts.append("<b>3 · Near 52-week low</b> ({})".format(len(near_lows)))
    if near_lows:
        rows = [f"{'SYM':<7} {'Rs':>7} {'rng%':>4}  {'52w lo':>7} {'hi':>7}"]
        rows.append("-" * 40)
        for s in near_lows:
            sym = str(s.get("symbol", "?"))[:7]
            ltp = _fmt_num(s.get("ltp"), 1)
            p = int(round(_pct_from_52w_low(s)))
            lo = _fmt_num(s.get("week_52_low"), 0)
            hi = _fmt_num(s.get("week_52_high"), 0)
            rows.append(f"{sym:<7} {ltp:>7} {p:>4}  {lo:>7} {hi:>7}")
        parts.append(_mono_block(rows))
    else:
        parts.append("<i>None.</i>")

    parts.extend(["", "<i>Not financial advice.</i>"])
    return "\n".join(parts)


def send_telegram(message: str, chat_id: str | None, parse_mode: str = "HTML") -> None:
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4096
    chunks = [message[i:i + max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
            },
        )
        if not resp.ok:
            logger.error(
                "Telegram send failed chat=%s: %s %s",
                chat_id,
                resp.status_code,
                resp.text,
            )
        else:
            logger.info("Telegram sent → %s", chat_id)


if __name__ == "__main__":
    buys, sells, near_lows, n_screened = get_classified_stocks()

    llm_output = ""
    if buys:
        llm_output = get_llm_picks(buys)
        logger.info("LLM output:\n%s", llm_output)

    send_group = TELEGRAM_SEND_TO in ("group", "both")
    send_dm = TELEGRAM_SEND_TO in ("dm", "both")

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping Telegram")
    else:
        logger.info("TELEGRAM_SEND_TO=%s", TELEGRAM_SEND_TO)

        if send_group:
            if TELEGRAM_CHAT_ID:
                group_body = format_group_digest(buys, near_lows, llm_output or "")
                send_telegram(group_body, chat_id=TELEGRAM_CHAT_ID)
            else:
                logger.warning(
                    "TELEGRAM_SEND_TO includes group but TELEGRAM_CHAT_ID is unset"
                )

        if send_dm:
            if TELEGRAM_DM_CHAT_ID:
                detail = format_personal_report(
                    llm_output or "",
                    buys,
                    near_lows,
                    n_screened,
                )
                send_telegram(detail, chat_id=TELEGRAM_DM_CHAT_ID)
            else:
                logger.warning(
                    "TELEGRAM_SEND_TO includes dm but TELEGRAM_DM_CHAT_ID is unset"
                )
