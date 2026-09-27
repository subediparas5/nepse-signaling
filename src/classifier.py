"""
Walk-forward learned classifier: P(stock beats the same-day universe average over 20 days).

Inputs are the stock features as per-date percentile ranks (stationary across market levels)
plus market-regime features. Two model families are trained on a schedule, each time only
on rows whose 20-day outcome had fully played out before the retrain date:

  logit  L2 logistic regression — simple, naturally close to calibrated
  gbm    shallow, heavily regularised histogram gradient boosting — can learn interactions
         such as "momentum helps only when breadth is strong"

Probabilities are evaluated for calibration (does 0.6 mean 60%?) and for ranking against
the step-4 score in backtest.py. Whether they are used live is decided by that comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STOCK_FEATURES = [
    "ret_5d", "ret_20d", "ret_60d", "dist_sma20", "dist_sma50", "dist_sma120", "rsi14",
    "macd_hist_pct", "range_pos", "drawdown_120", "atr14_pct", "vol_20d", "rvol_20",
    "rturnover_20", "sector_rel_20d", "abs_sector_rel_1d", "turnover_med_20",
]
REGIME_FEATURES = ["index_ret_20d", "breadth_sma50", "index_vol_20d"]

TARGET_H = 20
RETRAIN_EVERY = 20
MIN_TRAIN_DATES = 60
MODEL_NAMES = ("logit", "gbm")


def make_model(name: str):
    if name == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))
    if name == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=400,
            l2_regularization=5.0, random_state=0,
        )
    raise ValueError(name)


def design_matrix(df: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile ranks of stock features (missing -> 0.5) plus that date's regime inputs."""
    dates = df.index.get_level_values("date")
    src = df.assign(abs_sector_rel_1d=df["sector_rel_1d"].abs())
    X = pd.DataFrame(index=df.index)
    for f in STOCK_FEATURES:
        X[f] = src[f].groupby(dates).rank(pct=True).fillna(0.5)
    reg = regime.reindex(dates)
    for f in REGIME_FEATURES:
        col = reg[f].astype(float).to_numpy()
        X[f] = np.where(np.isnan(col), np.nanmedian(col) if np.isfinite(col).any() else 0.0, col)
    return X


def target(df: pd.DataFrame, h: int = TARGET_H) -> pd.Series:
    ex = df[f"fwd_excess_{h}"]
    return (ex > 0).astype(float).where(ex.notna())


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    calendar: pd.Index,
    models: tuple[str, ...] = MODEL_NAMES,
    h: int = TARGET_H,
    retrain_every: int = RETRAIN_EVERY,
    min_train_dates: int = MIN_TRAIN_DATES,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Out-of-sample probabilities for every row. A model retrained at calendar position t sees
    only rows dated d with pos(d) + 1 + h <= pos(t) and predicts dates t .. t+retrain_every-1.
    """
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    row_pos = pos.reindex(X.index.get_level_values("date")).to_numpy()
    preds = pd.DataFrame(np.nan, index=X.index, columns=list(models))
    retrains: list[str] = []
    for t in range(len(calendar)):
        if (t - (min_train_dates + 1 + h)) % retrain_every != 0 or t < min_train_dates + 1 + h:
            continue
        train = (row_pos + 1 + h <= t) & y.notna().to_numpy()
        test = (row_pos >= t) & (row_pos < t + retrain_every)
        if not test.any() or len(np.unique(y[train])) < 2:
            continue
        retrains.append(calendar[t])
        for name in models:
            m = make_model(name).fit(X[train], y[train])
            preds.loc[test, name] = m.predict_proba(X[test])[:, 1]
    return preds, retrains


def fit_latest(X: pd.DataFrame, y: pd.Series, calendar: pd.Index, name: str, h: int = TARGET_H):
    """Model trained on every realised outcome as of the last calendar date (for live use)."""
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    row_pos = pos.reindex(X.index.get_level_values("date")).to_numpy()
    train = (row_pos + 1 + h <= len(calendar) - 1) & y.notna().to_numpy()
    return make_model(name).fit(X[train], y[train])


def calibration_table(p: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised hit rate by probability decile."""
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), realised=("y", "mean"))


def brier(p: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    return float(((d["p"] - d["y"]) ** 2).mean())
