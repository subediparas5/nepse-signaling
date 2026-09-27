import numpy as np
import pandas as pd
import pytest

import regime as R
import scoring as S


def _calendar(n):
    return pd.Index([f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)], name="date")


def _ic_frame(cal, value=0.1):
    return pd.DataFrame({k: value for k in S.COMPONENTS}, index=cal)


def test_walk_forward_weights_use_only_realised_outcomes():
    cal = _calendar(120)
    ic = _ic_frame(cal)
    base = S.walk_forward_weights(ic, cal, horizon=20, min_dates=10)

    t = 80
    tampered = ic.copy()
    # ICs dated t-20 .. end have outcomes finishing after t, so they must not affect weights at t.
    tampered.iloc[t - 20:, :] = -5.0
    after = S.walk_forward_weights(tampered, cal, horizon=20, min_dates=10)
    pd.testing.assert_series_equal(base.iloc[t], after.iloc[t])
    assert not after.iloc[t + 1].equals(base.iloc[t + 1])


def test_walk_forward_weights_fall_back_to_prior_and_normalise():
    cal = _calendar(30)
    w = S.walk_forward_weights(_ic_frame(cal), cal, horizon=20, min_dates=40)
    prior = pd.Series(S.PRIOR_WEIGHTS) / sum(S.PRIOR_WEIGHTS.values())
    pd.testing.assert_series_equal(w.iloc[-1], prior[list(S.COMPONENTS)], check_names=False)
    assert w.abs().sum(axis=1).round(12).eq(1).all()


def test_component_without_enough_ics_keeps_prior_weight():
    cal = _calendar(100)
    ic = _ic_frame(cal, 0.2)
    ic["trend"] = np.nan  # e.g. SMA120 still warming up
    w = S.walk_forward_weights(ic, cal, horizon=5, min_dates=10).iloc[-1]
    assert w.notna().all() and w["trend"] > 0


def _long(values: dict[str, list[float]], dates=("d1",)):
    idx = pd.MultiIndex.from_product([list(dates), [f"S{i}" for i in range(len(next(iter(values.values()))))]],
                                     names=["date", "symbol"])
    return pd.DataFrame({k: v * len(dates) for k, v in values.items()}, index=idx)


def test_overall_score_negative_weight_reverses_component():
    comp = _long({k: [0.2, 0.8] for k in S.COMPONENTS})
    comp["n_components"] = 5
    up = S.overall_score(comp, pd.Series({k: 0.2 for k in S.COMPONENTS}))
    down = S.overall_score(comp, pd.Series({k: -0.2 for k in S.COMPONENTS}))
    assert list(up.round(6)) == [20.0, 80.0]
    assert list(down.round(6)) == [80.0, 20.0]


def test_overall_score_needs_min_components():
    comp = _long({k: [0.5] for k in S.COMPONENTS})
    comp[["trend", "momentum", "low_risk"]] = np.nan
    comp["n_components"] = 2
    assert S.overall_score(comp, pd.Series(S.PRIOR_WEIGHTS)).isna().all()


def test_classify_thresholds_and_overrides():
    n = 10
    score = _long({"s": [float(i) for i in range(n)]})["s"]
    risk = pd.Series(50.0, index=score.index)
    risk.iloc[8] = 95  # would be STRONG_SETUP by score, but too volatile
    turnover = pd.Series(5e6, index=score.index)
    turnover.iloc[7] = 1e5  # illiquid
    comps = pd.Series(5, index=score.index)
    new = pd.Series(False, index=score.index)
    new.iloc[6] = True
    out = S.classify(score, risk, turnover, comps, new).tolist()
    assert out[9] == "STRONG_SETUP"
    assert out[8] == "HIGH_RISK"
    assert out[7] == "AVOID"
    assert out[6] == "INSUFFICIENT_DATA"
    assert out[0] == "AVOID" and out[1] == "NEUTRAL"  # below the 20th percentile only
    assert out[5] == "WATCH" and out[3] == "NEUTRAL"


def test_component_scores_orient_low_risk_and_calm():
    df = _long({
        "dist_sma50": [0.1, -0.1], "dist_sma120": [0.1, -0.1], "drawdown_120": [0.0, -0.3], "range_pos": [0.9, 0.1],
        "ret_20d": [0.05, -0.05], "ret_60d": [0.1, -0.1], "vol_20d": [0.01, 0.05], "atr14_pct": [0.01, 0.05],
        "sector_rel_1d": [0.001, -0.08], "turnover_med_20": [5e7, 1e6],
    })
    comp = S.component_scores(df)
    assert (comp.iloc[0][list(S.COMPONENTS)] > comp.iloc[1][list(S.COMPONENTS)]).all()
    assert (comp["n_components"] == 5).all()


def test_regime_labels():
    n = 120
    dates = _calendar(n)
    up = pd.Series(np.linspace(1000, 1500, n), index=dates)
    down = pd.Series(np.linspace(1500, 1000, n), index=dates)
    stocks_up = pd.DataFrame(0.05, index=dates, columns=["A", "B", "C"])
    stocks_down = -stocks_up
    assert R.compute_regime(up, stocks_up)["regime"].iloc[-1] == "BULLISH"
    assert R.compute_regime(down, stocks_down)["regime"].iloc[-1] == "BEARISH"
    # Index rising but breadth weak -> not bullish
    assert R.compute_regime(up, stocks_down)["regime"].iloc[-1] == "NEUTRAL"
    assert R.compute_regime(up, stocks_up)["regime"].iloc[10] == "NEUTRAL"  # SMA50 warm-up
    assert R.compute_regime(up, stocks_up)["breadth_sma50"].iloc[-1] == pytest.approx(1.0)
