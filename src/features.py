"""
Point-in-time features and forward-return labels from the committed history in data/.

Everything is computed on wide (date x symbol) frames aligned to the NEPSE index trading
calendar, so a day a stock did not trade still counts as a trading day: its close is
carried forward, volume/turnover/trades are 0, and `traded` is False.

Features at date t use only rows <= t. Labels (`fwd_*`) look forward and must never be
fed back into a rule or feature — `tests/test_features.py` checks both properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import history_store

HORIZONS = (1, 5, 10, 20)

# Stored 52w fields exist only from daily snapshots; history uses a trailing range instead.
RANGE_WINDOW = 240
RANGE_MIN_OBS = 120

# NEPSE daily band is ±10%; a day whose low is already this far up was locked limit-up.
LIMIT_UP_LOCK = 0.09

# Fresh listings behave differently (a few run up many-fold); keep them out of the main universe.
NEW_LISTING_SESSIONS = 120

_NUMERIC = [
    "open", "high", "low", "close", "prev_close", "vwap",
    "volume", "turnover", "trades", "week_52_high", "week_52_low",
]


@dataclass
class Panel:
    """Wide frames indexed by date (rows) x symbol (columns)."""

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    open: pd.DataFrame
    vwap: pd.DataFrame
    volume: pd.DataFrame
    turnover: pd.DataFrame
    trades: pd.DataFrame
    traded: pd.DataFrame
    sector: pd.Series
    index_close: pd.Series


def build_panel(prices: pd.DataFrame, index: pd.DataFrame, sectors: dict[str, str]) -> Panel:
    """`prices`/`index` are long frames with a `date` column (see history_store fields)."""
    prices = prices.copy()
    for c in _NUMERIC:
        if c in prices:
            prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices = prices[prices["close"] > 0]

    calendar = pd.Index(sorted(set(index["date"]) | set(prices["date"])), name="date")
    calendar = calendar[calendar >= prices["date"].min()]

    symbols = sorted(prices["symbol"].unique())

    def wide(col: str) -> pd.DataFrame:
        if col not in prices:
            return pd.DataFrame(np.nan, index=calendar, columns=symbols)
        return prices.pivot(index="date", columns="symbol", values=col).reindex(index=calendar, columns=symbols)

    close_raw = wide("close")
    traded = close_raw.notna()
    close = close_raw.ffill()
    listed = close.notna()  # False before a symbol's first trade

    def fill_price(raw: pd.DataFrame) -> pd.DataFrame:
        return raw.where(traded, close)

    def fill_zero(raw: pd.DataFrame) -> pd.DataFrame:
        return raw.fillna(0).where(listed)

    idx = index.copy()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    index_close = idx.set_index("date")["close"].reindex(calendar).ffill()

    return Panel(
        close=close,
        high=fill_price(wide("high")),
        low=fill_price(wide("low")),
        open=wide("open"),
        vwap=wide("vwap"),
        volume=fill_zero(wide("volume")),
        turnover=fill_zero(wide("turnover")),
        trades=fill_zero(wide("trades")),
        traded=traded,
        sector=pd.Series(sectors).reindex(close.columns),
        index_close=index_close,
    )


def load_panel(data_dir: Path = history_store.DATA_DIR) -> Panel:
    prices = pd.DataFrame(history_store.load_prices(data_dir))
    index = pd.DataFrame(history_store.load_index(data_dir))
    secs = {r["symbol"]: r["sector"] for r in history_store._read(data_dir / "securities.csv")}
    return build_panel(prices, index, secs)


def _wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _sector_demean(frame: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """Subtract each date's sector median (cross-sectional); NaN for stocks with no sector."""
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    known = sector.dropna()
    for _, cols in known.groupby(known).groups.items():
        cols = list(cols)
        out[cols] = frame[cols].sub(frame[cols].median(axis=1), axis=0)
    return out


def compute_features(p: Panel) -> dict[str, pd.DataFrame]:
    """Wide feature frames. Each value at t depends only on rows <= t."""
    c = p.close
    prev = c.shift(1)
    ret_1d = c / prev - 1
    f: dict[str, pd.DataFrame] = {"ret_1d": ret_1d}

    for n in (5, 20, 60):
        f[f"ret_{n}d"] = c / c.shift(n) - 1
    for n in (20, 50, 120, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n, min_periods=n).mean() - 1

    delta = c.diff()
    avg_gain = _wilder(delta.clip(lower=0), 14)
    avg_loss = _wilder(-delta.clip(upper=0), 14)
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0).mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    f["rsi14"] = rsi

    tr = np.maximum(np.maximum(p.high - p.low, (p.high - prev).abs()), (p.low - prev).abs())
    f["atr14_pct"] = _wilder(tr, 14) / c

    f["vol_20d"] = ret_1d.rolling(20, min_periods=20).std()

    macd = c.ewm(span=12, adjust=False, min_periods=26).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c

    # Relative participation vs the *previous* 20 sessions, so today is not in its own baseline.
    f["rvol_20"] = (p.volume / p.volume.shift(1).rolling(20, min_periods=20).mean()).replace(np.inf, np.nan)
    f["rturnover_20"] = (p.turnover / p.turnover.shift(1).rolling(20, min_periods=20).median()).replace(
        np.inf, np.nan
    )
    f["turnover_med_20"] = p.turnover.rolling(20, min_periods=20).median()

    hi = p.high.rolling(RANGE_WINDOW, min_periods=RANGE_MIN_OBS).max()
    lo = p.low.rolling(RANGE_WINDOW, min_periods=RANGE_MIN_OBS).min()
    f["range_hi"], f["range_lo"] = hi, lo
    f["range_pos"] = ((c - lo) / (hi - lo)).where(hi > lo)
    f["drawdown_120"] = c / c.rolling(120, min_periods=60).max() - 1

    # Symbols already trading on the first stored day are treated as seasoned.
    first_day = p.traded.iloc[0] if len(p.traded) else pd.Series(dtype=bool)
    f["new_listing"] = (p.traded.cumsum() <= NEW_LISTING_SESSIONS) & ~first_day

    f["sector_rel_1d"] = _sector_demean(ret_1d, p.sector)
    f["sector_rel_20d"] = _sector_demean(f["ret_20d"], p.sector)
    idx_ret_20 = p.index_close / p.index_close.shift(20) - 1
    f["market_rel_20d"] = f["ret_20d"].sub(idx_ret_20, axis=0)
    return f


def forward_labels(p: Panel, horizons: tuple[int, ...] = HORIZONS) -> dict[str, pd.DataFrame]:
    """
    Enter at the close of t+1 (the 09:00 run sees t's data; you trade during t+1), exit at
    the close of t+1+h. No entry unless the stock traded on t (a signal day) and on t+1
    without being locked at the upper limit.
    """
    c = p.close
    entry = c.shift(-1)
    locked = p.low.shift(-1) >= c * (1 + LIMIT_UP_LOCK)
    fillable = p.traded & p.traded.shift(-1, fill_value=False) & ~locked
    entry = entry.where(fillable)

    out: dict[str, pd.DataFrame] = {"fillable": fillable}
    for h in horizons:
        out[f"fwd_ret_{h}"] = c.shift(-(1 + h)) / entry - 1
        # Worst low between entry and exit (t+2 .. t+1+h), relative to entry.
        out[f"fwd_mdd_{h}"] = (p.low.rolling(h, min_periods=h).min().shift(-(1 + h)) / entry - 1).clip(upper=0)
    return out


def add_excess(long: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """
    `fwd_excess_h` = forward return minus the same-date equal-weight mean *of the rows given*,
    so a falling or rising market does not masquerade as signal. Filter the universe first.
    """
    out = long.copy()
    dates = out.index.get_level_values("date")
    for h in horizons:
        col = out[f"fwd_ret_{h}"]
        out[f"fwd_excess_{h}"] = col - col.groupby(dates).transform("mean")
    return out


def to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack wide frames into one long frame indexed by (date, symbol)."""
    return pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
