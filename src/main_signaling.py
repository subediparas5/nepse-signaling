import html
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import events
import history_store
import scoring
from nepse_official import (
    TRADABLE_SECTORS,
    get_business_date,
    get_index_history,
    get_price_adjustments,
    get_official_listed_stocks,
    get_official_share_price_lookup,
)

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
    "rsi14", "vol_20d", "atr14_pct", "turnover_med_20",
    "flagged_since", "sessions", "since_ret", "recent_scores",
]

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

SYSTEM_PROMPT = """\
You are a Nepal stock market analyst writing short notes for a retail trader. You receive the \
top-ranked NEPSE stocks from a quantitative model. You do NOT pick, drop or re-rank them — you \
explain what the numbers show and what could go wrong.

The model ranks stocks against each other each day using official exchange prices (adjusted for \
bonus/rights). In its backtest, steady uptrends (price above 50/120-day averages, shallow \
drawdowns), low volatility and no one-day shock outperformed; volatile names and names near \
52-week lows lagged. Scores are 0-100 percentiles within today's market; trend, momentum, \
low_risk, calm and liquidity are 0-1 ranks. Returns and distances are fractions (0.05 = 5%). \
turnover_med_20 is the 20-day median daily turnover in Rs. flagged_since / sessions / since_ret \
describe how long the stock has been a setup and its return since then; recent_scores are the \
last few daily scores. There is no P/E, EPS or news data: do not invent fundamentals, news or \
price targets.

Return ONLY a JSON object, no prose and no code fences, in exactly this shape:
{"notes": [{"symbol": "ABC", "summary": "<= 20 words on what the numbers show",
            "risks": ["<= 12 words each, at most 3"]}]}
Include every symbol you were given, once each, in the same order.\
"""

LLM_SUMMARY_MAX = 160
LLM_RISK_MAX = 90
LLM_RISKS = 3


def fetch_market_snapshot() -> tuple[list[dict], list[dict]]:
    """Returns (today's market fields per tradable stock, listed_stocks).

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

    return pre_stocks, listed_stocks


SCORE_FIELDS = [
    "classification", "score", "opportunity_score", "risk_score",
    "trend", "momentum", "low_risk", "calm", "liquidity",
    "ret_20d", "ret_60d", "dist_sma50", "dist_sma120", "drawdown_120", "range_pos",
    "rsi14", "vol_20d", "atr14_pct", "turnover_med_20",
    "flagged_since", "sessions", "since_ret", "recent_scores",
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


def get_llm_notes(candidates: list[dict], *, log_label: str = "SETUPS") -> str:
    """Raw DeepSeek reply for `candidates` ('' when skipped). Parse with `parse_llm_notes`."""
    if not candidates:
        logger.warning("get_llm_notes(%s): empty candidate list — skipping API", log_label)
        return ""
    if not OPEN_AI_API_KEY:
        logger.warning("get_llm_notes(%s): OPEN_AI_API_KEY not set — skipping DeepSeek", log_label)
        return ""
    client = OpenAI(api_key=OPEN_AI_API_KEY, base_url="https://api.deepseek.com")
    payload = compact_for_llm(candidates)
    logger.info("Calling DeepSeek %s (%s) with %s candidates", DEEPSEEK_MODEL, log_label, len(payload))
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
        )
    except Exception:
        # Notes are optional; the digest goes out without them.
        logger.exception("DeepSeek call failed")
        return ""
    return response.choices[0].message.content or ""


def _clip(text: object, limit: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def parse_llm_notes(raw: str, allowed_symbols: list[str]) -> dict[str, dict]:
    """
    Validate DeepSeek's JSON: keep only notes for symbols we sent, clip lengths, drop anything
    malformed. Returns {symbol: {"summary": str, "risks": [str]}} in `allowed_symbols` order.
    """
    if not raw or not raw.strip():
        return {}
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        logger.warning("DeepSeek reply has no JSON object")
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("DeepSeek reply is not valid JSON")
        return {}
    notes = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(notes, list):
        return {}
    allowed = set(allowed_symbols)
    out: dict[str, dict] = {}
    for n in notes:
        if not isinstance(n, dict):
            continue
        sym = str(n.get("symbol", "")).strip().upper()
        summary = n.get("summary")
        if sym not in allowed or sym in out or not isinstance(summary, str) or not summary.strip():
            continue
        risks = n.get("risks") if isinstance(n.get("risks"), list) else []
        out[sym] = {
            "summary": _clip(summary, LLM_SUMMARY_MAX),
            "risks": [_clip(r, LLM_RISK_MAX) for r in risks if isinstance(r, str) and r.strip()][:LLM_RISKS],
        }
    dropped = len(notes) - len(out)
    if dropped:
        logger.warning("Dropped %s invalid/unknown DeepSeek notes", dropped)
    return {sym: out[sym] for sym in allowed_symbols if sym in out}


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


def _record_line(label: str, rec: dict | None) -> str | None:
    if not rec or rec.get("n", 0) < 100:
        return None
    return (
        f"{label}: {rec['hit'] * 100:.0f}% beat the market over {rec['horizon']}d "
        f"(avg {rec['excess'] * 100:+.1f}%, n={rec['n']:,})"
    )


def format_telegram_digest(
    notes: dict[str, dict],
    setups: list[dict],
    context: dict | None,
    class_counts: dict[str, int],
) -> str:
    """Telegram HTML: regime, setups table with signal age, track record, DeepSeek notes, dropped names."""
    context = context or {}
    ts = html.escape(_npt_now(), quote=False)
    parts = [f"<b>NEPSE</b> · <code>{ts}</code>", format_market_line(context or None), ""]

    n_strong = class_counts.get("STRONG_SETUP", 0)
    n_setup = class_counts.get("SETUP", 0)
    parts.append(f"<b>Top setups</b> ({n_strong} strong, {n_setup} setup)")
    if setups:
        rows = [f"{'SYM':<7} {'Rs':>7} {'scr':>3} {'rsk':>3} {'20d%':>5} {'ses':>3} {'since':>6}", "-" * 41]
        for s in setups:
            sym = str(s.get("symbol", "?"))[:7] + ("*" if s.get("classification") == "STRONG_SETUP" else "")
            ltp = _fmt_num(s.get("ltp"), 1)
            scr = int(round(float(s.get("score") or 0)))
            rsk = int(round(float(s.get("risk_score") or 0)))
            ses = s.get("sessions")
            ses_txt = str(int(ses)) if ses is not None else "—"
            rows.append(
                f"{sym:<8}{ltp:>7} {scr:>3} {rsk:>3} {_pct_cell(s.get('ret_20d')):>5} {ses_txt:>3} "
                f"{_pct_cell(s.get('since_ret')):>6}"
            )
        parts.append(_mono_block(rows))
    else:
        parts.append("<i>None.</i>")

    records = [
        _record_line(f"Strong setups, last {context.get('recent_sessions', 90)} sessions",
                     (context.get("track_record_recent") or {}).get("STRONG_SETUP")),
        _record_line("all history", (context.get("track_record") or {}).get("STRONG_SETUP")),
    ]
    records = [r for r in records if r]
    if records:
        parts.append("<i>" + html.escape(" · ".join(records), quote=False) + "</i>")

    if notes:
        parts.extend(["", "<b>Notes</b> (DeepSeek, explains the ranking — does not change it)"])
        for sym, n in notes.items():
            line = f"<b>{html.escape(sym)}</b> {html.escape(n['summary'], quote=False)}"
            if n["risks"]:
                line += " <i>Risk: " + html.escape("; ".join(n["risks"]), quote=False) + "</i>"
            parts.append(line)

    dropped = context.get("dropped") or []
    if dropped:
        shown = ", ".join(f"{d['symbol']}→{d['now'].lower().replace('_', ' ')}" for d in dropped[:10])
        more = f" +{len(dropped) - 10} more" if len(dropped) > 10 else ""
        parts.extend(["", f"<b>No longer a setup</b>: {html.escape(shown, quote=False)}{more}"])

    flagged = ", ".join(
        f"{class_counts[c]} {c.lower().replace('_', ' ')}" for c in ("HIGH_RISK", "AVOID") if class_counts.get(c)
    )
    if flagged:
        parts.append(f"<i>Flagged: {flagged}.</i>")
    parts.extend([
        "",
        "<i>* strong setup. scr = rank vs today's market (0-100), rsk = volatility rank, ses = sessions "
        "flagged, since = return since first flagged. Relative ranking, not a price forecast. "
        "Not financial advice.</i>",
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


def send_telegram(message: str, chat_id: str | None, parse_mode: str = "HTML") -> bool:
    """Send `message` (split into Telegram-sized chunks). True only if every chunk was accepted."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4096
    chunks = (
        _telegram_html_chunks(message, max_len)
        if parse_mode == "HTML"
        else [message[i : i + max_len] for i in range(0, len(message), max_len)]
    )
    ok = True
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
            ok = False
            logger.error(
                "Telegram send failed chat=%s: %s %s",
                chat_id,
                resp.status_code,
                resp.text,
            )
        else:
            logger.info("Telegram sent → %s", chat_id)
    return ok


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
    """Record every stock's model classification and scores for later evaluation."""
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
            "close": s.get("close") if s.get("close") is not None else s.get("ltp"),
        }
        for s in all_stocks
    ]
    logger.info("History %s: %s signal rows changed", business_date, history_store.upsert_signals(rows, data_dir))


if __name__ == "__main__":
    business_date = get_business_date()
    # The workflow has a backup schedule; a business date gets one digest (also skips holidays,
    # when the business date does not change).
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and history_store.digest_sent(business_date):
        logger.info("Digest for %s already sent — nothing to do", business_date)
        sys.exit(0)
    all_stocks, listed_stocks = fetch_market_snapshot()
    try:
        persist_market_snapshot(business_date, all_stocks, listed_stocks)
        history_store.upsert_index(get_index_history())
        history_store.upsert_corporate_actions(get_price_adjustments())
        history_store.upsert_events(events.fetch_events())
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
    logger.info("Classification mix: %s", class_counts)
    try:
        persist_signals(business_date, all_stocks, context)
    except Exception:
        logger.exception("Failed to persist signals")

    setups = select_setups(all_stocks)
    raw_notes = get_llm_notes(setups)
    notes = parse_llm_notes(raw_notes, [s["symbol"] for s in setups])
    if raw_notes:
        logger.info("DeepSeek notes: %s of %s setups", len(notes), len(setups))

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping Telegram")
    elif not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID is unset")
    elif send_telegram(format_telegram_digest(notes, setups, context, class_counts), chat_id=TELEGRAM_CHAT_ID):
        history_store.record_digest_sent(business_date, datetime.now(timezone.utc).isoformat(timespec="seconds"))
