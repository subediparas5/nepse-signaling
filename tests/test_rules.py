import pytest

from nepse_signal_rules import (
    _dividend_yield_vote,
    _has_fundamentals,
    _liquidity_vote,
    _week52_vote,
    classify_nepse_signal,
    stock_eps,
)


def _flat_stock(**overrides):
    """A price-only stock that triggers no votes: mid 52w range, flat day, at VWAP."""
    base = {
        "ltp": 100.0,
        "close": 100.0,
        "open": 100.0,
        "prev_close": 100.0,
        "vwap": 100.0,
        "week_52_high": 150.0,
        "week_52_low": 50.0,
        "turnover": 1_000_000.0,
        "transactions": 80.0,
        "range_pct": 1.0,
        "diff_pct": 0.0,
        "_sector_median_diff": 0.0,
    }
    base.update(overrides)
    return base


def test_flat_stock_is_hold_with_zero_scores():
    r = classify_nepse_signal(_flat_stock(), "BANKING")
    assert r["signal_verdict"] == "HOLD"
    assert (r["signal_buy_score"], r["signal_sell_score"]) == (0, 0)
    assert r["signal_confidence"] == 0


def test_price_only_buy_threshold_is_4_with_margin_3():
    # At 52w low (+2), bullish close above open (+1), above VWAP (+1) -> buy 4, sell 0
    s = _flat_stock(ltp=52.0, close=52.0, open=50.5, prev_close=50.0, vwap=51.0)
    r = classify_nepse_signal(s, "HYDROPOWER")
    assert r["signal_buy_score"] == 4
    assert r["signal_sell_score"] == 0
    assert r["signal_verdict"] == "BUY"
    assert r["signal_confidence"] == 100


def test_price_only_lean_buy_at_3():
    # At 52w low (+2), above VWAP (+1) -> buy 3
    s = _flat_stock(ltp=52.0, close=52.0, open=52.0, prev_close=52.0, vwap=51.0)
    r = classify_nepse_signal(s, "HYDROPOWER")
    assert r["signal_buy_score"] == 3
    assert r["signal_verdict"] == "LEAN_BUY"


def test_fundamentals_present_raises_thresholds_even_without_votes():
    # P/E inside the neutral band casts no vote but still means "fundamentals present".
    s = _flat_stock(ltp=52.0, close=52.0, open=50.5, prev_close=50.0, vwap=51.0, pe=20.0)
    r = classify_nepse_signal(s, "BANKING")
    assert r["signal_fundamental_buy"] == r["signal_fundamental_sell"] == 0
    assert r["signal_buy_score"] == 4
    assert r["signal_verdict"] == "LEAN_BUY"  # would be BUY price-only


def test_ipo_when_ma120_zero():
    assert classify_nepse_signal(_flat_stock(ma120=0), "BANKING")["signal_verdict"] == "IPO"


@pytest.mark.parametrize(
    "ltp, expected",
    [
        (55, (2, 0)),   # 5% of range
        (70, (1, 0)),   # 20%
        (100, (0, 0)),  # 50%
        (125, (0, 1)),  # 75%
        (135, (0, 2)),  # 85%
        (148, (0, 3)),  # 98%
    ],
)
def test_week52_vote_bands(ltp, expected):
    b, s, _ = _week52_vote(ltp, 150, 50)
    assert (b, s) == expected


def test_week52_vote_ignores_bad_range():
    assert _week52_vote(100, 50, 50) == (0, 0, None)
    assert _week52_vote(None, 150, 50) == (0, 0, None)


@pytest.mark.parametrize(
    "turnover, tx, expected",
    [
        (400_000, 500, (0, 1)),
        (1_000_000, 30, (0, 1)),
        (6_000_000, 150, (1, 0)),
        (1_000_000, 80, (0, 0)),
        (None, None, (0, 0)),
    ],
)
def test_liquidity_vote(turnover, tx, expected):
    b, s, _ = _liquidity_vote(turnover, tx)
    assert (b, s) == expected


def test_stock_eps_prefers_ttm_and_keeps_zero():
    assert stock_eps({"eps_ttm": 0, "eps": 5}) == 0.0
    assert stock_eps({"eps": "12.5"}) == 12.5
    assert stock_eps({}) is None


def test_dividend_vote_penalizes_no_dividend_with_positive_eps():
    assert _dividend_yield_vote(100, 0, 10)[:2] == (0, 1)
    assert _dividend_yield_vote(100, 5, 10)[:2] == (1, 0)


def test_dividend_vote_reads_eps_ttm_via_classify():
    # Only eps_ttm is set; the "no dividend despite positive EPS" penalty must still apply.
    r = classify_nepse_signal(_flat_stock(dpps=0, eps_ttm=10), "HYDROPOWER")
    assert r["signal_fundamental_sell"] == 1
    assert "No dividend despite positive EPS" in r["signal_reasons"]


def test_has_fundamentals_ignores_none_and_garbage():
    assert not _has_fundamentals({"pe": None, "promoter_percentage": None})
    assert not _has_fundamentals({"pe": "n/a"})
    assert _has_fundamentals({"npl": "2.1"})
