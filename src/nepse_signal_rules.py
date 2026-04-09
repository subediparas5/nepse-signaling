from __future__ import annotations

from typing import Any

TRADABLE_SECTORS = frozenset({
    "BANKING", "DEVBANK", "FINANCE", "MICROFINANCE",
    "HYDROPOWER", "MANUFACTURE", "HOTELS", "TRADING",
    "LIFEINSU", "NONLIFEINSU", "INVESTMENT", "OTHERS",
})

BFI_SECTORS = frozenset(
    {"BANKING", "DEVBANK", "FINANCE", "MICROFINANCE", "LIFEINSU", "NONLIFEINSU"}
)
BANK_LIKE_SECTORS = frozenset({"BANKING", "DEVBANK", "FINANCE", "MICROFINANCE"})

PE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "BANKING":      (15, 30),
    "DEVBANK":      (12, 25),
    "FINANCE":      (12, 25),
    "MICROFINANCE": (10, 20),
    "LIFEINSU":     (15, 35),
    "NONLIFEINSU":  (12, 25),
    "HYDROPOWER":   (20, 40),
    "MANUFACTURE":  (15, 30),
    "HOTELS":       (15, 30),
    "INVESTMENT":   (12, 25),
    "TRADING":      (12, 25),
    "OTHERS":       (15, 30),
}
DEFAULT_PE = (15, 30)


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pct(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None

def _ma_position_vote(
    ltp: Any, ma120: Any, ma180: Any,
) -> tuple[int, int, str | None]:
    """Weight 2: Price vs 120d/180d MAs — trend-following signal."""
    px = _to_float(ltp)
    m120 = _to_float(ma120)
    m180 = _to_float(ma180)
    if px is None:
        return 0, 0, None

    below_both = (
        m120 is not None and m180 is not None
        and px < m120 and px < m180
    )
    above_both = (
        m120 is not None and m180 is not None
        and px > m120 and px > m180
    )

    if below_both:
        pct_below = min(
            (m120 - px) / m120 * 100 if m120 else 0,
            (m180 - px) / m180 * 100 if m180 else 0,
        )
        if pct_below > 5:
            return 2, 0, f"Below both MAs ({pct_below:.0f}% under — value zone)"
        return 1, 0, "Below both MAs (near support)"

    if above_both:
        pct_above = min(
            (px - m120) / m120 * 100 if m120 else 0,
            (px - m180) / m180 * 100 if m180 else 0,
        )
        if pct_above > 15:
            return 0, 2, f"Above both MAs by {pct_above:.0f}% (overextended)"
        if pct_above > 8:
            return 0, 1, f"Above both MAs by {pct_above:.0f}%"

    if m120 is not None and px < m120:
        return 1, 0, "Below 120d MA"
    if m120 is not None and px > m120 * 1.15:
        return 0, 1, "Well above 120d MA"

    return 0, 0, None


def _week52_vote(
    ltp: Any, high52: Any, low52: Any, turnover: Any = None,
) -> tuple[int, int, str | None]:
    """Weight 1-2: 52-week range position with breakout/capitulation detection."""
    px = _to_float(ltp)
    hi = _to_float(high52)
    lo = _to_float(low52)
    if px is None or hi is None or lo is None or hi <= lo:
        return 0, 0, None

    pct_from_low = (px - lo) / (hi - lo)
    vol = _to_float(turnover)
    has_volume = vol is not None and vol > 5_000_000

    if pct_from_low >= 0.98:
        if has_volume:
            return 2, 0, f"52w high breakout on volume (Rs {vol/1e6:.1f}M)"
        return 1, 0, "52w high breakout (momentum)"

    if pct_from_low <= 0.02:
        if has_volume:
            return 2, 0, f"At 52w low with volume (Rs {vol/1e6:.1f}M) — capitulation buy?"
        return 1, 0, "At 52w low (potential capitulation/value)"

    if pct_from_low <= 0.20:
        return 1, 0, f"Near 52w low ({pct_from_low:.0%} of range)"
    if pct_from_low >= 0.85:
        return 0, 1, f"Near 52w high ({pct_from_low:.0%} of range)"
    return 0, 0, None


def _daily_momentum_vote(
    close: Any, prev_close: Any, open_price: Any,
) -> tuple[int, int, str | None]:
    """Weight 1: Yesterday's price action — close vs prev_close and open."""
    cl = _to_float(close)
    pc = _to_float(prev_close)
    op = _to_float(open_price)
    if cl is None or pc is None or pc == 0:
        return 0, 0, None

    day_chg_pct = (cl - pc) / pc * 100

    closed_above_open = op is not None and cl > op

    if day_chg_pct > 3 and closed_above_open:
        return 1, 0, f"Bullish day (+{day_chg_pct:.1f}%, closed above open)"
    if day_chg_pct < -3 and op is not None and cl < op:
        return 0, 1, f"Bearish day ({day_chg_pct:.1f}%, closed below open)"
    return 0, 0, None


def _vwap_vote(ltp: Any, vwap: Any) -> tuple[int, int, str | None]:
    """Weight 1: Close vs VWAP — institutional buying/selling pressure."""
    px = _to_float(ltp)
    vw = _to_float(vwap)
    if px is None or vw is None or vw == 0:
        return 0, 0, None
    diff_pct = (px - vw) / vw * 100
    if diff_pct > 1.5:
        return 1, 0, f"Closed above VWAP (+{diff_pct:.1f}%)"
    if diff_pct < -1.5:
        return 0, 1, f"Closed below VWAP ({diff_pct:.1f}%)"
    return 0, 0, None


def _liquidity_vote(
    turnover: Any, transactions: Any,
) -> tuple[int, int, str | None]:
    """Weight 1: Turnover + transaction count as liquidity/interest proxy."""
    to = _to_float(turnover)
    tx = _to_float(transactions)
    if to is None and tx is None:
        return 0, 0, None

    if to is not None and to < 500_000:
        return 0, 1, f"Low turnover Rs {to/1e6:.2f}M (illiquid)"
    if tx is not None and tx < 50:
        return 0, 1, f"Only {int(tx)} transactions (illiquid)"

    if to is not None and to > 10_000_000 and tx is not None and tx > 500:
        return 1, 0, f"High activity: Rs {to/1e6:.1f}M, {int(tx)} txns"
    return 0, 0, None


def _ma_convergence_vote(
    ltp: Any, ma120: Any, ma180: Any,
) -> tuple[int, int, str | None]:
    """Weight 1: 120d/180d MA convergence — potential crossover signal."""
    px = _to_float(ltp)
    m120 = _to_float(ma120)
    m180 = _to_float(ma180)
    if px is None or m120 is None or m180 is None or m180 == 0:
        return 0, 0, None

    gap_pct = abs(m120 - m180) / m180 * 100
    if gap_pct > 2:
        return 0, 0, None

    if px > m120 and px > m180:
        return 1, 0, f"MAs converging ({gap_pct:.1f}% gap), price above — golden cross setup"
    if px < m120 and px < m180:
        return 0, 1, f"MAs converging ({gap_pct:.1f}% gap), price below — death cross risk"
    return 0, 0, None


def _range_confirmation_vote(
    close: Any, open_price: Any, range_pct: Any,
) -> tuple[int, int, str | None]:
    """Weight 1: Wide intraday range + directional close = conviction."""
    cl = _to_float(close)
    op = _to_float(open_price)
    rp = _to_float(range_pct)
    if cl is None or op is None or rp is None:
        return 0, 0, None

    if rp > 3 and cl > op:
        return 1, 0, f"Wide range ({rp:.1f}%) bullish bar"
    if rp > 3 and cl < op:
        return 0, 1, f"Wide range ({rp:.1f}%) bearish bar"
    return 0, 0, None


# ---------------------------------------------------------------------------
# Fundamental votes — from nepsetrading recent-report + stocks-listed
# ---------------------------------------------------------------------------

def _eps_qoq_vote(q1: Any, q2: Any) -> tuple[int, int, str | None]:
    a1, a2 = _parse_pct(q1), _parse_pct(q2)
    if a1 is None or a2 is None:
        return 0, 0, None
    if a1 > a2:
        return 1, 0, "EPS QoQ improving"
    if a1 < a2:
        return 0, 1, "EPS QoQ weakening"
    return 0, 0, None


def _dividend_yield_vote(
    ltp: Any, dpps: Any, eps: Any,
) -> tuple[int, int, str | None]:
    """Weight 1: Dividend yield — NEPSE investors love dividends."""
    px = _to_float(ltp)
    dps = _to_float(dpps)
    if px is None or dps is None or px <= 0:
        return 0, 0, None

    dy = dps / px * 100
    if dy > 4:
        return 1, 0, f"Dividend yield {dy:.1f}% (attractive)"
    if dy >= 2:
        return 1, 0, f"Dividend yield {dy:.1f}%"

    eps_val = _to_float(eps)
    if dps == 0 and eps_val is not None and eps_val > 0:
        return 0, 1, "No dividend despite positive EPS"
    return 0, 0, None


def _fundamental_votes(stock: dict[str, Any], sector: str) -> tuple[int, int, list[str]]:
    b, s = 0, 0
    reasons: list[str] = []

    pe_lo, pe_hi = PE_THRESHOLDS.get(sector, DEFAULT_PE)
    pe = _to_float(stock.get("pr_ratio")) or _to_float(stock.get("pe"))
    if pe is not None:
        if pe < 0:
            s += 2
            reasons.append("Negative P/E (loss-making)")
        elif 0 < pe < pe_lo:
            b += 1
            reasons.append(f"P/E {pe:.1f} (below {pe_lo} for {sector})")
        elif pe > pe_hi:
            s += 1
            reasons.append(f"P/E {pe:.1f} (above {pe_hi} for {sector})")

    eps = _to_float(stock.get("eps_ttm"))
    if eps is not None and eps < 0:
        s += 2
        reasons.append(f"Negative EPS ({eps:.2f})")

    if sector in BFI_SECTORS:
        pb = _to_float(stock.get("pb"))
        if pb is not None and pb > 0:
            if pb < 1.5:
                b += 1
                reasons.append(f"P/B {pb:.2f} (cheap for BFI)")
            elif pb > 3.0:
                s += 1
                reasons.append(f"P/B {pb:.2f} (expensive for BFI)")

    promoter = _to_float(stock.get("promoter_percentage"))
    if promoter is not None:
        if promoter > 51:
            b += 1
            reasons.append(f"Promoter {promoter:.0f}%")
        elif promoter < 25:
            s += 1
            reasons.append(f"Low promoter {promoter:.0f}%")

    eb, es, er = _eps_qoq_vote(stock.get("q1"), stock.get("q2"))
    b += eb
    s += es
    if er:
        reasons.append(er)

    db, ds, dr = _dividend_yield_vote(
        stock.get("ltp"), stock.get("dpps"), stock.get("eps"),
    )
    b += db
    s += ds
    if dr:
        reasons.append(dr)

    if sector in BANK_LIKE_SECTORS:
        roe = _to_float(stock.get("roe_ttm")) or _to_float(stock.get("return_on_equity"))
        if roe is None:
            roe = _to_float(stock.get("roe"))
        if roe is not None:
            if roe > 15:
                b += 1
                reasons.append(f"ROE {roe:.1f}%")
            elif roe < 10:
                s += 1
                reasons.append(f"ROE {roe:.1f}% (weak)")

        npl = _to_float(stock.get("npl_to_total_loan")) or _to_float(stock.get("npl"))
        if npl is not None:
            if npl < 3:
                b += 1
                reasons.append(f"NPL {npl:.1f}% (healthy)")
            elif npl > 5:
                s += 1
                reasons.append(f"NPL {npl:.1f}% (risky)")

    return b, s, reasons


def _technical_votes(stock: dict[str, Any]) -> tuple[int, int, list[str]]:
    b, s = 0, 0
    reasons: list[str] = []

    ltp = stock.get("ltp") or stock.get("close")

    for fn, args in (
        (_ma_position_vote, (ltp, stock.get("ma120"), stock.get("ma180"))),
        (_week52_vote, (ltp, stock.get("week_52_high"), stock.get("week_52_low"), stock.get("turnover"))),
        (_daily_momentum_vote, (stock.get("close"), stock.get("prev_close"), stock.get("open"))),
        (_vwap_vote, (ltp, stock.get("vwap"))),
        (_liquidity_vote, (stock.get("turnover"), stock.get("transactions"))),
        (_ma_convergence_vote, (ltp, stock.get("ma120"), stock.get("ma180"))),
        (_range_confirmation_vote, (stock.get("close"), stock.get("open"), stock.get("range_pct"))),
    ):
        bb, ss, r = fn(*args)
        b += bb
        s += ss
        if r:
            reasons.append(r)

    return b, s, reasons


def classify_nepse_signal(stock: dict[str, Any], sector: str) -> dict[str, Any]:
    fb, fs, fr = _fundamental_votes(stock, sector)
    tb, ts, tr = _technical_votes(stock)

    buy_score = fb + tb
    sell_score = fs + ts
    reasons = fr + tr

    margin = 3
    if buy_score >= 6 and buy_score >= sell_score + margin:
        verdict = "BUY"
    elif sell_score >= 6 and sell_score >= buy_score + margin:
        verdict = "SELL"
    elif buy_score >= 4 and buy_score > sell_score:
        verdict = "LEAN_BUY"
    elif sell_score >= 4 and sell_score > buy_score:
        verdict = "LEAN_SELL"
    else:
        verdict = "HOLD"

    return {
        "signal_verdict": verdict,
        "signal_buy_score": buy_score,
        "signal_sell_score": sell_score,
        "signal_fundamental_buy": fb,
        "signal_fundamental_sell": fs,
        "signal_technical_buy": tb,
        "signal_technical_sell": ts,
        "signal_reasons": " | ".join(reasons),
    }
