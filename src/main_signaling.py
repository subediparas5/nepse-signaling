import html
import json
import logging
import os
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import history_store
import scoring
from nepse_official import (
    get_business_date,
    get_index_history,
    get_price_adjustments,
    get_official_listed_stocks,
    get_official_share_price_lookup,
)
from nepse_signal_rules import TRADABLE_SECTORS, classify_nepse_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: without it the rule engine and Telegram digest still run, minus the LLM section.
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


LLM_FIELDS = [
    "symbol", "sector", "ltp", "classification", "score", "opportunity_score", "risk_score",
    "trend", "momentum", "low_risk", "calm", "liquidity",
    "ret_20d", "ret_60d", "dist_sma50", "dist_sma120", "drawdown_120", "range_pos",
    "rsi14", "vol_20d", "atr14_pct", "turnover_med_20", "week_52_high", "week_52_low",
]

SYSTEM_PROMPT = """\
You are a Nepal stock market analyst writing short notes for a retail trader. You receive the \
top-ranked NEPSE stocks from a quantitative model; you do NOT pick or re-rank them.

The model ranks stocks against each other each day using official exchange prices. In its \
backtest, names with steady uptrends (price above 50/120-day averages, shallow drawdowns), low \
volatility and no one-day shock outperformed; volatile names and names near 52-week lows lagged. \
Scores are 0-100 percentiles within today's market; component values (trend, momentum, low_risk, \
calm, liquidity) are 0-1 ranks. Returns and distances are fractions (0.05 = 5%). \
turnover_med_20 is the 20-day median daily turnover in Rs. There is no P/E or EPS data.

For each stock write one line: what the numbers say, and the main risk to watch (e.g. extended \
above averages, thin turnover, sector concentration). Do not invent news, fundamentals or targets.

Output format — return ONLY this, nothing else:

SYMBOL | Rs PRICE | note (15 words max)

One stock per line, same order as given. No numbering, no markdown, no headers.\
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


def classify_all_stocks() -> tuple[list[dict], list[dict]]:
    """Returns (all_classified_stocks, listed_stocks).

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
    for data in pre_stocks:
        data["_sector_median_diff"] = sector_medians.get(data["sector"])
        data.update(classify_nepse_signal(data, data["sector"]))
        all_stocks.append(data)
    return all_stocks, listed_stocks


SCORE_FIELDS = [
    "classification", "score", "opportunity_score", "risk_score",
    "trend", "momentum", "low_risk", "calm", "liquidity",
    "ret_20d", "ret_60d", "dist_sma50", "dist_sma120", "drawdown_120", "range_pos",
    "rsi14", "vol_20d", "atr14_pct", "turnover_med_20",
]
TOP_SETUPS = 8


def attach_scores(all_stocks: list[dict], scored: pd.DataFrame) -> None:
    """Copy model fields onto each stock dict; unscored symbols become INSUFFICIENT_DATA."""
    for s in all_stocks:
        sym = s.get("symbol")
        if sym in scored.index:
            row = scored.loc[sym]
            for k in SCORE_FIELDS:
                v = row.get(k)
                s[k] = None if v is None or (isinstance(v, float) and pd.isna(v)) else v
        else:
            s["classification"] = "INSUFFICIENT_DATA"


def select_setups(all_stocks: list[dict], n: int = TOP_SETUPS) -> list[dict]:
    """Highest-scoring STRONG_SETUP names, topped up with SETUP if there are fewer than n."""
    ranked = sorted(
        (s for s in all_stocks if s.get("classification") in ("STRONG_SETUP", "SETUP") and s.get("score") is not None),
        key=lambda s: (s["classification"] != "STRONG_SETUP", -float(s["score"])),
    )
    return ranked[:n]


def compact_for_llm(candidates: list[dict]) -> list[dict]:
    def clean(v):
        return round(float(v), 4) if isinstance(v, float) else v

    return [
        {k: clean(s.get(k)) for k in LLM_FIELDS if s.get(k) is not None}
        for s in candidates
    ]


def get_llm_picks(
    candidates: list[dict],
    *,
    system_prompt: str | None = None,
    log_label: str = "BUY",
) -> str:
    if not candidates:
        logger.warning("get_llm_picks(%s): empty candidate list — skipping API", log_label)
        return ""
    if not OPEN_AI_API_KEY:
        logger.warning("get_llm_picks(%s): OPEN_AI_API_KEY not set — skipping DeepSeek", log_label)
        return ""
    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")
    payload = compact_for_llm(candidates)
    user_content = json.dumps(payload, default=str)
    logger.info("Calling DeepSeek (%s) with %s candidates", log_label, len(payload))

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
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


def _llm_mono_rows(raw: str) -> list[str]:
    """
    Monospace rows for <pre>: pipe-separated lines become a padded column layout;
    otherwise long lines are wrapped to fit phones.
    """
    lines = raw.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return []

    pipe_lines = [ln for ln in nonempty if "|" in ln]
    use_table = bool(pipe_lines) and (
        len(pipe_lines) >= 2
        or (len(pipe_lines) == 1 and pipe_lines[0].count("|") >= 2)
        or len(pipe_lines) * 2 >= len(nonempty)
    )

    if not use_table:
        width = 44
        out: list[str] = []
        for ln in lines:
            if not ln:
                out.append("")
                continue
            if len(ln) <= width:
                out.append(ln)
            else:
                wrapped = textwrap.wrap(
                    ln,
                    width=width,
                    replace_whitespace=False,
                    break_long_words=True,
                )
                out.extend(wrapped or [ln[:width]])
        return out

    max_cell = 24
    max_cols = 8
    cell_matrix: list[list[str]] = []
    for ln in lines:
        if "|" not in ln or not ln.strip():
            continue
        cells = [c.strip()[:max_cell] for c in ln.split("|")]
        cell_matrix.append(cells)
    ncols = min(max(len(r) for r in cell_matrix), max_cols) if cell_matrix else 1
    widths = [2] * ncols
    for r in cell_matrix:
        for i in range(ncols):
            cell = r[i] if i < len(r) else ""
            widths[i] = min(max(widths[i], len(cell)), max_cell)

    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        if "|" in ln:
            cells = [c.strip()[:max_cell] for c in ln.split("|")]
            while len(cells) < ncols:
                cells.append("")
            cells = cells[:ncols]
            out.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)))
        else:
            out.append(ln.rstrip())
    return out


def _mono_block(lines: list[str]) -> str:
    body = "\n".join(html.escape(line, quote=False) for line in lines)
    return f"<pre>{body}</pre>"


def _pct_cell(val, decimals=1) -> str:
    try:
        return f"{float(val) * 100:+.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_market_line(context: dict | None) -> str:
    if not context:
        return "<i>Market regime unavailable.</i>"
    parts = [f"<b>{html.escape(str(context.get('regime') or 'NEUTRAL'))}</b>"]
    if context.get("index_ret_20d") is not None:
        parts.append(f"NEPSE 20d {_pct_cell(context['index_ret_20d'])}%")
    if context.get("breadth_sma50") is not None:
        parts.append(f"{float(context['breadth_sma50']) * 100:.0f}% above SMA50")
    if context.get("high_vol"):
        parts.append("high volatility")
    return "Market: " + " · ".join(parts)


def format_telegram_digest(
    llm_output: str,
    setups: list[dict],
    context: dict | None,
    class_counts: dict[str, int],
) -> str:
    """Telegram HTML: market regime, optional LLM notes, top-ranked setups table, class counts."""
    ts = html.escape(_npt_now(), quote=False)
    parts = [f"<b>NEPSE</b> · <code>{ts}</code>", format_market_line(context), ""]

    raw = llm_output or ""
    if any(x.strip() for x in raw.splitlines()):
        parts.extend(["<b>LLM notes</b>", _mono_block(_llm_mono_rows(raw)), ""])

    n_strong = class_counts.get("STRONG_SETUP", 0)
    n_setup = class_counts.get("SETUP", 0)
    parts.append(f"<b>Top setups</b> ({n_strong} strong, {n_setup} setup)")
    rec = ((context or {}).get("track_record") or {}).get("STRONG_SETUP")
    if rec and rec.get("n", 0) >= 100:
        parts.append(
            f"<i>Past strong setups beat the market over {rec['horizon']}d in {rec['hit'] * 100:.0f}% of "
            f"{rec['n']:,} cases (avg {rec['excess'] * 100:+.1f}% vs market).</i>"
        )
    if setups:
        rows = [f"{'SYM':<7} {'Rs':>7} {'scr':>3} {'rsk':>3} {'20d%':>5}  cls", "-" * 36]
        for s in setups:
            sym = str(s.get("symbol", "?"))[:7]
            ltp = _fmt_num(s.get("ltp"), 1)
            scr = int(round(float(s.get("score") or 0)))
            rsk = int(round(float(s.get("risk_score") or 0)))
            cls = "S+" if s.get("classification") == "STRONG_SETUP" else "S"
            rows.append(f"{sym:<7} {ltp:>7} {scr:>3} {rsk:>3} {_pct_cell(s.get('ret_20d')):>5}  {cls}")
        parts.append(_mono_block(rows))
    else:
        parts.append("<i>None.</i>")

    flagged = ", ".join(
        f"{class_counts[c]} {c.lower().replace('_', ' ')}" for c in ("HIGH_RISK", "AVOID") if class_counts.get(c)
    )
    if flagged:
        parts.append(f"<i>Flagged: {flagged}.</i>")
    parts.extend([
        "",
        "<i>scr = rank vs today's market (0-100), rsk = volatility rank. "
        "Relative ranking, not a price forecast. Not financial advice.</i>",
    ])
    return "\n".join(parts)


def _split_oversized_pre_block(block: str, max_len: int) -> list[str]:
    """Telegram <pre> blocks must stay intact per message; split inner lines across messages."""
    open_t, close_t = "<pre>", "</pre>"
    if not (block.startswith(open_t) and block.endswith(close_t)):
        return [block[:max_len]] if len(block) > max_len else [block]
    inner = block[len(open_t) : -len(close_t)]
    lines = inner.split("\n")
    out: list[str] = []
    chunk_lines: list[str] = []
    overhead = len(open_t) + len(close_t)

    def flush() -> None:
        nonlocal chunk_lines
        if chunk_lines:
            out.append(open_t + "\n".join(chunk_lines) + close_t)
            chunk_lines = []

    for line in lines:
        candidate = "\n".join(chunk_lines + [line]) if chunk_lines else line
        if overhead + len(candidate) <= max_len:
            chunk_lines.append(line)
        else:
            flush()
            if overhead + len(line) <= max_len:
                chunk_lines = [line]
            else:
                for i in range(0, len(line), max_len - overhead - 1):
                    slice_ = line[i : i + max_len - overhead - 1]
                    out.append(open_t + slice_ + close_t)
                chunk_lines = []
    flush()
    return out or [open_t + close_t]


def _split_long_plain_html(frag: str, max_len: int) -> list[str]:
    """Split non-<pre> HTML on newlines so tags are not cut mid-token."""
    if len(frag) <= max_len:
        return [frag]
    out: list[str] = []
    buf = ""
    for line in frag.split("\n"):
        extra = len(line) + (1 if buf else 0)
        if len(buf) + extra > max_len and buf:
            out.append(buf)
            buf = ""
        if len(line) > max_len:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(line), max_len):
                out.append(line[i : i + max_len])
            continue
        buf = buf + "\n" + line if buf else line
    if buf:
        out.append(buf)
    return out


def _telegram_html_chunks(message: str, max_len: int = 4096) -> list[str]:
    """
    Split for Telegram sendMessage without breaking HTML.
    Naive fixed-size splits corrupt <pre>...</pre> and trigger parse errors.
    """
    if len(message) <= max_len:
        return [message]

    fragments: list[str] = []
    i = 0
    while i < len(message):
        j = message.find("<pre>", i)
        if j == -1:
            fragments.append(message[i:])
            break
        if j > i:
            fragments.append(message[i:j])
        k = message.find("</pre>", j)
        if k == -1:
            fragments.append(message[j:])
            break
        k += len("</pre>")
        fragments.append(message[j:k])
        i = k

    chunks: list[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            chunks.append(buf)
            buf = ""

    for frag in fragments:
        is_pre = frag.startswith("<pre>") and frag.endswith("</pre>")
        pieces = (
            _split_oversized_pre_block(frag, max_len)
            if is_pre and len(frag) > max_len
            else ([frag] if is_pre else _split_long_plain_html(frag, max_len))
        )
        for piece in pieces:
            if not piece:
                continue
            joiner = "\n" if buf else ""
            if len(buf) + len(joiner) + len(piece) <= max_len:
                buf = buf + joiner + piece if buf else piece
            else:
                flush_buf()
                if len(piece) > max_len:
                    if piece.startswith("<pre>"):
                        chunks.extend(_split_oversized_pre_block(piece, max_len))
                    else:
                        chunks.extend(_split_long_plain_html(piece, max_len))
                else:
                    buf = piece
    flush_buf()
    return chunks


def send_telegram(message: str, chat_id: str | None, parse_mode: str = "HTML") -> None:
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4096
    chunks = (
        _telegram_html_chunks(message, max_len)
        if parse_mode == "HTML"
        else [message[i : i + max_len] for i in range(0, len(message), max_len)]
    )
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


def persist_market_snapshot(
    business_date: str,
    all_stocks: list[dict],
    listed_stocks: list[dict],
    data_dir: Path = history_store.DATA_DIR,
) -> None:
    """Append today's market fields and the NEPSE index to data/ (see history_store)."""
    price_rows = [
        {
            **s,
            "date": business_date,
            "close": s.get("close") if s.get("close") is not None else s.get("ltp"),
            "trades": s.get("transactions"),
        }
        for s in all_stocks
        if s.get("ltp") is not None or s.get("close") is not None
    ]
    history_store.upsert_securities(listed_stocks, seen_on=business_date, data_dir=data_dir)
    logger.info("History %s: %s price rows changed", business_date, history_store.upsert_prices(price_rows, data_dir))


def persist_signals(
    business_date: str,
    all_stocks: list[dict],
    context: dict | None,
    data_dir: Path = history_store.DATA_DIR,
) -> None:
    """Record every stock's model classification and the legacy vote verdict for later evaluation."""
    regime = (context or {}).get("regime")
    rows = [
        {
            "date": business_date,
            "symbol": s["symbol"],
            "sector": s.get("sector"),
            "classification": s.get("classification"),
            "score": s.get("score"),
            "opportunity_score": s.get("opportunity_score"),
            "risk_score": s.get("risk_score"),
            "regime": regime,
            "verdict": s.get("signal_verdict"),
            "buy_score": s.get("signal_buy_score"),
            "sell_score": s.get("signal_sell_score"),
            "confidence": s.get("signal_confidence"),
            "technical_buy": s.get("signal_technical_buy"),
            "technical_sell": s.get("signal_technical_sell"),
            "fundamental_buy": s.get("signal_fundamental_buy"),
            "fundamental_sell": s.get("signal_fundamental_sell"),
            "close": s.get("close") if s.get("close") is not None else s.get("ltp"),
            "reasons": s.get("signal_reasons"),
        }
        for s in all_stocks
    ]
    logger.info("History %s: %s signal rows changed", business_date, history_store.upsert_signals(rows, data_dir))


if __name__ == "__main__":
    all_stocks, listed_stocks = classify_all_stocks()
    business_date = get_business_date()
    try:
        persist_market_snapshot(business_date, all_stocks, listed_stocks)
        history_store.upsert_index(get_index_history())
        history_store.upsert_corporate_actions(get_price_adjustments())
    except Exception:
        # History is for research; never let it block the daily digest.
        logger.exception("Failed to persist market snapshot")

    context: dict | None = None
    try:
        scored, context = scoring.score_latest()
        if context["date"] != business_date:
            logger.warning("Scored date %s differs from business date %s", context["date"], business_date)
        attach_scores(all_stocks, scored)
    except Exception:
        logger.exception("Scoring failed — digest will have no setups")
        for s in all_stocks:
            s["classification"] = "INSUFFICIENT_DATA"

    class_counts = dict(Counter(s.get("classification") for s in all_stocks))
    logger.info("Classification mix: %s · legacy verdicts: %s", class_counts,
                dict(Counter(s.get("signal_verdict") for s in all_stocks)))
    try:
        persist_signals(business_date, all_stocks, context)
    except Exception:
        logger.exception("Failed to persist signals")

    setups = select_setups(all_stocks)
    llm_output = get_llm_picks(setups, log_label="SETUPS") if setups else ""
    if llm_output:
        logger.info("LLM output:\n%s", llm_output)

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping Telegram")
    elif not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID is unset")
    else:
        send_telegram(format_telegram_digest(llm_output, setups, context, class_counts), chat_id=TELEGRAM_CHAT_ID)
