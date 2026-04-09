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


def _norm_str(x: Any) -> str:
    return str(x or "").strip().lower()


def _signal_buy_sell_from_api_text(s: str) -> tuple[int, int]:
    if not s or s in ("neutral", "-"):
        return 0, 0
    if ("below" in s and "30" in s) or "oversold" in s:
        return 1, 0
    if ("above" in s and "70" in s) or "overbought" in s:
        return 0, 1
    if "golden" in s or ("bull" in s and "bear" not in s):
        return 1, 0
    if "death" in s or ("bear" in s and "bull" not in s):
        return 0, 1
    tokens = s.replace("/", " ").split()
    has_by = "by" in tokens or s.rstrip().endswith(" by")
    has_sl = "sl" in tokens or s.rstrip().endswith(" sl")
    if has_sl and not has_by:
        return 0, 1
    if has_by and not has_sl:
        return 1, 0
    if s in ("buy", "strong buy"):
        return 1, 0
    if s in ("sell", "strong sell"):
        return 0, 1
    return 0, 0


def _rsi_vote(rsi_signal: Any) -> tuple[int, int, str | None]:
    s = _norm_str(rsi_signal)
    b, se = _signal_buy_sell_from_api_text(s)
    if b:
        return 2, 0, "RSI oversold/buy"
    if se:
        return 0, 2, "RSI overbought/sell"
    return 0, 0, None


# Weight 2: MACD (rare, meaningful crossover)
def _macd_vote(macd: Any) -> tuple[int, int, str | None]:
    s = _norm_str(macd)
    b, se = _signal_buy_sell_from_api_text(s)
    if b:
        return 2, 0, "MACD bullish crossover"
    if se:
        return 0, 1, "MACD bearish"
    return 0, 0, None


# Weight 1: MA (fires often, moderate signal)
def _ma_vote(ma: Any) -> tuple[int, int, str | None]:
    s = _norm_str(ma)
    b, se = _signal_buy_sell_from_api_text(s)
    if b:
        return 1, 0, "MA buy"
    if se:
        return 0, 1, "MA sell"
    return 0, 0, None


# Weight 1: Volume breakout (50% above avg — meaningful for NEPSE)
def _volume_vote(volume: Any, avg30: Any) -> tuple[int, int, str | None]:
    v = _to_float(volume)
    a = _to_float(avg30)
    if v is not None and a is not None and a > 0:
        if v >= 1.5 * a:
            return 1, 0, f"Volume breakout ({v/a:.1f}x avg)"
        if v <= 0.3 * a:
            return 0, 1, "Very low volume"
    return 0, 0, None


# Weight 1: 52-week position
def _week52_vote(ltp: Any, high52: Any, low52: Any) -> tuple[int, int, str | None]:
    px = _to_float(ltp)
    hi = _to_float(high52)
    lo = _to_float(low52)
    if px is None or hi is None or lo is None or hi <= lo:
        return 0, 0, None
    pct_from_low = (px - lo) / (hi - lo)
    if pct_from_low <= 0.15:
        return 1, 0, f"Near 52w low ({pct_from_low:.0%} of range)"
    if pct_from_low >= 0.90:
        return 0, 1, f"Near 52w high ({pct_from_low:.0%} of range)"
    return 0, 0, None


# Weight 1: Accumulation/Distribution trend (smart money flow)
def _ad_trend_vote(ad_trend: Any) -> tuple[int, int, str | None]:
    s = _norm_str(ad_trend)
    if s == "accumulation":
        return 1, 0, "Accumulation (smart money inflow)"
    if s == "distribution":
        return 0, 1, "Distribution (smart money outflow)"
    return 0, 0, None


# Weight 1: Momentum — both timeframes must agree
def _momentum_vote(one_month: Any, three_month: Any) -> tuple[int, int, str | None]:
    m1 = _parse_pct(one_month)
    m3 = _parse_pct(three_month)
    if m1 is None or m3 is None:
        return 0, 0, None
    if m1 > 5 and m3 > 10:
        return 1, 0, f"Strong momentum (+{m1:.0f}% 1m, +{m3:.0f}% 3m)"
    if m1 < -5 and m3 < -10:
        return 0, 1, f"Weak momentum ({m1:.0f}% 1m, {m3:.0f}% 3m)"
    return 0, 0, None


def _sr_vote(
    ltp: Any,
    sup_lo: Any,
    sup_hi: Any,
    res_lo: Any,
    res_hi: Any,
) -> tuple[int, int, list[str]]:
    px = _to_float(ltp)
    reasons: list[str] = []
    b, s = 0, 0
    slo, shi = _to_float(sup_lo), _to_float(sup_hi)
    rlo, rhi = _to_float(res_lo), _to_float(res_hi)
    if px is not None and slo is not None and shi is not None and slo <= shi:
        if slo <= px <= shi:
            b += 1
            reasons.append("Price in support zone")
    if px is not None and rlo is not None and rhi is not None and rlo <= rhi:
        if rlo <= px <= rhi:
            s += 1
            reasons.append("Price in resistance zone")
    return b, s, reasons


def _eps_qoq_vote(q1: Any, q2: Any) -> tuple[int, int, str | None]:
    a1, a2 = _parse_pct(q1), _parse_pct(q2)
    if a1 is None or a2 is None:
        return 0, 0, None
    if a1 > a2:
        return 1, 0, "EPS QoQ improving"
    if a1 < a2:
        return 0, 1, "EPS QoQ weakening"
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

    for fn, key in (
        (_rsi_vote, "rsi_signal"),
        (_macd_vote, "macd_signal"),
        (_ma_vote, "ma_signal"),
    ):
        bb, ss, r = fn(stock.get(key))
        b += bb
        s += ss
        if r:
            reasons.append(r)

    bb, ss, r = _volume_vote(stock.get("volume"), stock.get("avg_volume_30_days"))
    b += bb
    s += ss
    if r:
        reasons.append(r)

    bb, ss, r = _week52_vote(
        stock.get("latesttransactionprice"),
        stock.get("week_52_high"),
        stock.get("week_52_low"),
    )
    b += bb
    s += ss
    if r:
        reasons.append(r)

    bb, ss, r = _ad_trend_vote(stock.get("ad_trend"))
    b += bb
    s += ss
    if r:
        reasons.append(r)

    bb, ss, r = _momentum_vote(
        stock.get("one_month_perf"),
        stock.get("three_month_perf"),
    )
    b += bb
    s += ss
    if r:
        reasons.append(r)

    bb, ss, rs = _sr_vote(
        stock.get("latesttransactionprice"),
        stock.get("support_zone_lower"),
        stock.get("support_zone_upper"),
        stock.get("resistance_zone_lower"),
        stock.get("resistance_zone_upper"),
    )
    b += bb
    s += ss
    reasons.extend(rs)

    return b, s, reasons


def classify_nepse_signal(stock: dict[str, Any], sector: str) -> dict[str, Any]:
    fb, fs, fr = _fundamental_votes(stock, sector)
    tb, ts, tr = _technical_votes(stock)

    buy_score = fb + tb
    sell_score = fs + ts
    reasons = fr + tr

    margin = 3
    if buy_score >= 5 and buy_score >= sell_score + margin:
        verdict = "BUY"
    elif sell_score >= 5 and sell_score >= buy_score + margin:
        verdict = "SELL"
    elif buy_score >= 3 and buy_score > sell_score:
        verdict = "LEAN_BUY"
    elif sell_score >= 3 and sell_score > buy_score:
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
