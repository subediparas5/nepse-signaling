import numpy as np
import pandas as pd

import backtest as B
import history_store as hs


def _write_history(data_dir, n_days=150, n_syms=40, seed=3):
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2025-10-01", periods=n_days).strftime("%Y-%m-%d"))
    rows = []
    for j in range(n_syms):
        c = 100.0
        for d in dates:
            c *= 1 + rng.normal(0, 0.02)
            rows.append({"date": d, "symbol": f"S{j:02d}", "close": c, "high": c * 1.01, "low": c * 0.99,
                         "volume": 1000, "turnover": 1000 * c * rng.uniform(1, 20), "trades": rng.integers(20, 200)})
    hs.upsert_prices(rows, data_dir)
    hs.upsert_index([{"date": d, "close": 2500 + i} for i, d in enumerate(dates)], data_dir)
    hs.upsert_securities(
        [{"symbol": f"S{j:02d}", "security_id": j, "sector": "BANKING" if j % 2 else "HYDROPOWER"}
         for j in range(n_syms)],
        dates[-1], data_dir,
    )


def test_backtest_runs_end_to_end_and_baseline_excess_is_zero(tmp_path):
    _write_history(tmp_path)
    report = B.run(tmp_path)
    assert "## Feature information coefficients" in report

    assert "## Scoring model (walk-forward)" in report and "Legacy" not in report

    df = B.F.build_dataset(B.F.load_panel(tmp_path))
    everything = df["close"].notna()
    s = B.group_stats(df, everything, 5, df.index.get_level_values("date")[len(df) // 2])
    assert abs(s["excess"]) < 1e-12
    # Trailing range needs >= RANGE_MIN_OBS sessions
    early = df.index.get_level_values("date") < sorted(set(df.index.get_level_values("date")))[B.F.RANGE_MIN_OBS - 1]
    assert df.loc[early, "range_pos"].isna().all()
