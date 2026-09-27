import numpy as np
import pandas as pd
import pytest

import classifier as C


def _xy(n_dates=140, n_syms=40, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.Index([f"d{i:03d}" for i in range(n_dates)], name="date")
    idx = pd.MultiIndex.from_product([cal, [f"S{j}" for j in range(n_syms)]], names=["date", "symbol"])
    X = pd.DataFrame(rng.uniform(size=(len(idx), 3)), index=idx, columns=["a", "b", "c"])
    y = pd.Series((X["a"] + rng.normal(0, 0.3, len(idx)) > 0.5).astype(float), index=idx)
    return X, y, cal


def test_walk_forward_training_never_sees_unrealised_outcomes():
    X, y, cal = _xy()
    h, every, min_dates = 5, 10, 20
    base, retrains = C.walk_forward_predict(X, y, cal, models=("logit",), h=h, retrain_every=every,
                                            min_train_dates=min_dates)
    assert retrains
    t = list(cal).index(retrains[1])
    # Flip every label whose outcome is not yet known at retrain position t.
    pos = pd.Series(np.arange(len(cal)), index=cal).reindex(X.index.get_level_values("date")).to_numpy()
    tampered = y.copy()
    tampered[pos + 1 + h > t] = 1 - tampered[pos + 1 + h > t]
    after, _ = C.walk_forward_predict(X, tampered, cal, models=("logit",), h=h, retrain_every=every,
                                      min_train_dates=min_dates)
    block = (pos >= t) & (pos < t + every)
    np.testing.assert_allclose(base.loc[block, "logit"], after.loc[block, "logit"])


def test_predictions_only_after_warm_up_and_informative():
    X, y, cal = _xy()
    preds, retrains = C.walk_forward_predict(X, y, cal, h=5, retrain_every=10, min_train_dates=20)
    first = list(cal).index(retrains[0])
    dates = X.index.get_level_values("date")
    assert preds[dates.isin(cal[:first])].isna().all().all()
    oos = preds.dropna()
    assert oos["logit"].corr(X.loc[oos.index, "a"]) > 0.8


def test_calibration_and_brier():
    p = pd.Series(np.linspace(0.05, 0.95, 1000))
    y = pd.Series((np.random.default_rng(1).uniform(size=1000) < p).astype(float))
    tab = C.calibration_table(p, y, bins=5)
    assert len(tab) == 5 and tab["n"].sum() == 1000
    assert (tab["realised"].diff().dropna() > 0).all()
    assert C.brier(p, y) < C.brier(pd.Series(0.5, index=y.index), y)
    assert C.brier(pd.Series([1.0, 0.0]), pd.Series([1.0, 0.0])) == pytest.approx(0.0)


def test_design_matrix_ranks_per_date_and_fills():
    idx = pd.MultiIndex.from_product([["d1", "d2"], ["A", "B"]], names=["date", "symbol"])
    df = pd.DataFrame({f: [1.0, 2.0, 5.0, np.nan] for f in C.STOCK_FEATURES if f != "abs_sector_rel_1d"}, index=idx)
    df["sector_rel_1d"] = [0.01, -0.02, 0.0, 0.0]
    reg = pd.DataFrame({"index_ret_20d": [0.01, np.nan], "breadth_sma50": [0.4, 0.6], "index_vol_20d": [0.01, 0.02]},
                       index=["d1", "d2"])
    X = C.design_matrix(df, reg)
    assert list(X["ret_20d"]) == [0.5, 1.0, 1.0, 0.5]  # d2/B missing -> neutral 0.5
    assert X.loc[("d2", "A"), "index_ret_20d"] == pytest.approx(0.01)  # NaN regime -> median
    assert X.loc[("d1", "B"), "abs_sector_rel_1d"] == 1.0
