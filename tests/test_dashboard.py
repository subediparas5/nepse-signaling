import json

import numpy as np
import pandas as pd
import pytest

import build_dashboard as D
import features as F


def _panel(closes):
    n = len(next(iter(closes.values())))
    dates = [f"2026-01-{i + 1:02d}" for i in range(n)]
    rows = [{"date": dates[i], "symbol": s, "close": c, "high": c, "low": c,
             "volume": 1.0, "turnover": c, "trades": 1.0}
            for s, cs in closes.items() for i, c in enumerate(cs)]
    index = pd.DataFrame({"date": dates, "close": [1000.0] * n})
    return F.build_panel(pd.DataFrame(rows), index, {s: "BANKING" for s in closes}), dates


def test_staggered_hold_one_slice_matches_hand_calc():
    # A doubles over the hold window after entry; hold=2 so the slice is 1/2 of capital.
    p, d = _panel({"A": [100, 100, 110, 121, 121, 121]})
    curve = D._staggered_hold(p, {d[0]: ["A"]}, d, hold=2, cost=0.0)
    # Signal d0 -> buy close d1 -> held d2, d3 (+10%, +10%) at half weight.
    assert curve[1] == pytest.approx(100.0)
    assert curve[2] == pytest.approx(105.0)
    assert curve[3] == pytest.approx(105.0 * 1.05)
    assert curve[-1] == pytest.approx(curve[3])


def test_staggered_hold_charges_one_round_trip_per_slice():
    p, d = _panel({"A": [100.0] * 6})
    curve = D._staggered_hold(p, {d[0]: ["A"]}, d, hold=2, cost=0.01)
    assert curve[-1] == pytest.approx(100 * (1 - 0.005) * (1 - 0.005), abs=1e-3)  # curves stored to 3 dp


def test_daily_rebalanced_turnover_and_cost():
    idx = pd.MultiIndex.from_product([["d1", "d2"], ["A", "B"]], names=["date", "symbol"])
    df = pd.DataFrame({"fwd_ret_1": [0.01, 0.01, 0.02, 0.02]}, index=idx)
    mask = pd.Series([True, False, False, True], index=idx)  # A then B: full switch
    curve, turn = D._daily_rebalanced(df, mask, ["d1", "d2"], cost=0.01)
    assert turn == pytest.approx(1.0)
    assert curve[0] == pytest.approx(100 * (1 + 0.01 - 0.02))
    assert curve[1] == pytest.approx(curve[0] * (1 + 0.02 - 0.02))


def test_render_embeds_json_safely(tmp_path, monkeypatch):
    tpl = tmp_path / "t.html"
    tpl.write_text("<script>const DATA = /*__DASHBOARD_DATA__*/null;</script>", encoding="utf-8")
    monkeypatch.setattr(D, "TEMPLATE", tpl)
    html = D.render({"x": "</script><b>", "n": 1})
    assert "</script><b>" not in html
    payload = html.split("const DATA = ", 1)[1].rsplit(";</script>", 1)[0]
    assert json.loads(payload.replace("<\\/", "</")) == {"x": "</script><b>", "n": 1}


def test_render_rejects_nan_and_missing_marker(tmp_path, monkeypatch):
    tpl = tmp_path / "t.html"
    tpl.write_text("no marker", encoding="utf-8")
    monkeypatch.setattr(D, "TEMPLATE", tpl)
    with pytest.raises(RuntimeError):
        D.render({})
    tpl.write_text("/*__DASHBOARD_DATA__*/null", encoding="utf-8")
    with pytest.raises(ValueError):
        D.render({"bad": float("nan")})


def test_round_helper():
    assert D._r(np.float64(1.234567)) == 1.2346
    assert D._r(float("nan")) is None and D._r(None) is None and D._r("x") == "x"
