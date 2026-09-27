import numpy as np
import pandas as pd
import pytest

import features as F


def _panel(closes: dict[str, list[float | None]], lows: dict[str, list[float | None]] | None = None, sectors=None):
    """Long price/index frames from per-symbol close lists (None = no trade that day)."""
    n = len(next(iter(closes.values())))
    dates = [f"2026-01-{i + 1:02d}" if i < 31 else f"2026-02-{i - 30:02d}" for i in range(n)]
    rows = []
    for sym, cs in closes.items():
        for i, c in enumerate(cs):
            if c is None:
                continue
            lo = (lows or {}).get(sym, [None] * n)[i]
            rows.append({"date": dates[i], "symbol": sym, "close": c, "high": c * 1.01,
                         "low": lo if lo is not None else c * 0.99, "volume": 100.0, "turnover": 100.0 * c,
                         "trades": 10.0})
    index = pd.DataFrame({"date": dates, "close": [1000.0 + 3 * i + (i % 7) for i in range(n)]})
    secs = sectors or {s: "BANKING" for s in closes}
    return F.build_panel(pd.DataFrame(rows), index, secs)


def _random_panel(n_days=80, n_syms=6, seed=1):
    rng = np.random.default_rng(seed)
    closes = {
        f"S{j}": list(100 * np.cumprod(1 + rng.normal(0, 0.02, n_days))) for j in range(n_syms)
    }
    closes["S0"][10] = None  # a no-trade day
    return closes


def test_features_have_no_lookahead():
    closes = _random_panel()
    full = F.compute_features(_panel(closes))
    k = 50
    trunc = F.compute_features(_panel({s: c[:k] for s, c in closes.items()}))
    for name, frame in trunc.items():
        pd.testing.assert_frame_equal(
            frame.astype(float), full[name].iloc[:k].astype(float), check_names=False, obj=name
        )


def test_no_trade_day_carries_close_and_zeroes_volume():
    p = _panel(_random_panel())
    d = p.close.index[10]
    assert not p.traded.loc[d, "S0"]
    assert p.close.loc[d, "S0"] == p.close.iloc[9]["S0"]
    assert p.volume.loc[d, "S0"] == 0


def test_rsi_extremes():
    up = F.compute_features(_panel({"UP": [100 + i for i in range(30)], "FLAT": [100.0] * 30}))
    assert up["rsi14"]["UP"].iloc[-1] == pytest.approx(100)
    assert up["rsi14"]["FLAT"].iloc[-1] == pytest.approx(50)
    assert up["rsi14"]["UP"].iloc[:13].isna().all()


def test_forward_labels_entry_is_next_close():
    closes = {"A": [100, 110, 121, 133.1, 146.41, 161.051, 177.1561]}
    lab = F.forward_labels(_panel(closes), horizons=(1, 2))
    # Signal day 0 -> enter at day1 close 110, exit day2 close 121 (h=1) / day3 (h=2)
    assert lab["fwd_ret_1"]["A"].iloc[0] == pytest.approx(0.10)
    assert lab["fwd_ret_2"]["A"].iloc[0] == pytest.approx(0.21)
    assert np.isnan(lab["fwd_ret_1"]["A"].iloc[-2])  # exit beyond data


def test_forward_labels_skip_locked_limit_up_and_no_trade():
    closes = {"A": [100, 110, 111, 112, 113], "B": [100, None, 100, 100, 100]}
    lows = {"A": [99, 109.8, 110, 111, 112]}  # day1 traded only at ~+10%: locked
    p = _panel(closes, lows=lows)
    p.high.loc[p.high.index[1], "A"] = 110.0
    lab = F.forward_labels(p, horizons=(1,))
    assert not lab["fillable"]["A"].iloc[0]
    assert np.isnan(lab["fwd_ret_1"]["A"].iloc[0])
    assert lab["fillable"]["A"].iloc[1]
    assert not lab["fillable"]["B"].iloc[0]  # B did not trade on day1


def test_big_up_day_with_range_is_still_fillable():
    closes = {"A": [100, 112, 113]}  # +12% day that traded 109.5-113: buyable under a 15% band
    p = _panel(closes, lows={"A": [99, 109.5, 112]})
    p.high.loc[p.high.index[1], "A"] = 113.0
    assert F.forward_labels(p, horizons=(1,))["fillable"]["A"].iloc[0]


def test_add_excess_demeans_within_given_rows():
    p = _panel(_random_panel(30, 5))
    long = F.to_long(F.forward_labels(p, horizons=(1,)))
    ex = F.add_excess(long, horizons=(1,))["fwd_excess_1"]
    per_date = ex.groupby(level="date").mean().dropna()
    assert (per_date.abs() < 1e-12).all()


def test_new_listing_flag():
    closes = {"OLD": [100.0] * 10, "NEW": [None] * 4 + [50.0] * 6}
    f = F.compute_features(_panel(closes))
    assert not f["new_listing"]["OLD"].any()
    assert f["new_listing"]["NEW"].iloc[4:].all()


def test_sector_relative_uses_sector_median():
    closes = {"A": [100, 110], "B": [100, 100], "C": [100, 90], "H": [100, 150]}
    secs = {"A": "BANKING", "B": "BANKING", "C": "BANKING", "H": "HYDROPOWER"}
    f = F.compute_features(_panel(closes, sectors=secs))
    rel = f["sector_rel_1d"].iloc[-1]
    assert rel["A"] == pytest.approx(0.10) and rel["C"] == pytest.approx(-0.10)
    assert rel["H"] == pytest.approx(0.0)


def test_bonus_adjustment_removes_artificial_drop():
    # 100% bonus announced after day 2's close: day 3 trades at half price.
    closes = {"A": [100.0, 102.0, 104.0, 52.5, 53.0], "B": [50.0] * 5}
    p_raw = _panel(closes)
    d = list(p_raw.close.index)
    actions = pd.DataFrame([{"date": d[2], "symbol": "A", "factor": 0.5}])
    rows = []
    for sym, cs in closes.items():
        for i, c in enumerate(cs):
            rows.append({"date": d[i], "symbol": sym, "close": c, "high": c * 1.01, "low": c * 0.99,
                         "volume": 100.0, "turnover": 100.0 * c, "trades": 10.0})
    index = pd.DataFrame({"date": d, "close": [1000.0 + i for i in range(5)]})
    p = F.build_panel(pd.DataFrame(rows), index, {"A": "BANKING", "B": "BANKING"}, actions)
    r = p.close["A"] / p.close["A"].shift(1) - 1
    assert r.iloc[3] == pytest.approx(52.5 / 52.0 - 1)  # not -49.5%
    assert p.close["A"].iloc[2] == pytest.approx(52.0) and p.close["A"].iloc[4] == pytest.approx(53.0)
    assert p.volume["A"].iloc[0] == pytest.approx(200.0)  # share counts scale inversely
    assert p.turnover["A"].iloc[0] == pytest.approx(10_000.0)  # rupee turnover unchanged
    pd.testing.assert_series_equal(p.close["B"], p_raw.close["B"])


def test_actions_outside_window_or_unknown_symbol_are_ignored():
    closes = {"A": [100.0, 101.0, 102.0]}
    traded = _panel(closes).traded
    acts = pd.DataFrame([
        {"date": "2020-01-01", "symbol": "A", "factor": 0.5},
        {"date": traded.index[1], "symbol": "ZZZ", "factor": 0.5},
    ])
    assert (F.adjustment_factors(acts, traded) == 1.0).all().all()


def test_snapshot_features_need_open_and_vwap():
    p = _panel({"A": [100.0, 102.0, 104.0]})
    f = F.compute_features(p)
    assert f["gap_1d"]["A"].isna().all() and f["close_vs_vwap"]["A"].isna().all()
    p.open.loc[p.open.index[2], "A"] = 103.0
    p.vwap.loc[p.vwap.index[2], "A"] = 103.5
    f = F.compute_features(p)
    assert f["gap_1d"]["A"].iloc[2] == pytest.approx(103 / 102 - 1)
    assert f["close_vs_open"]["A"].iloc[2] == pytest.approx(104 / 103 - 1)
    assert f["close_vs_vwap"]["A"].iloc[2] == pytest.approx(104 / 103.5 - 1)
