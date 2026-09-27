"""
NEPSE market regime per date, from the index and cross-sectional breadth.

BULLISH  index > SMA50, SMA20 > SMA50, and at least half of stocks above their own SMA50
BEARISH  index < SMA50, SMA20 < SMA50, and fewer than half above their SMA50
NEUTRAL  anything else (including warm-up days before SMA50 exists)

`high_vol` flags days whose 20-day index volatility exceeds 1.5x its trailing 120-day median.
Only data up to each date is used.
"""

from __future__ import annotations

import pandas as pd

BREADTH_LINE = 0.5
HIGH_VOL_MULT = 1.5


def compute_regime(index_close: pd.Series, dist_sma50: pd.DataFrame) -> pd.DataFrame:
    """
    `index_close`: NEPSE close by date. `dist_sma50`: wide stock frame (close / SMA50 - 1),
    restricted by the caller to the universe that should count toward breadth.
    """
    c = index_close.astype(float)
    sma20 = c.rolling(20, min_periods=20).mean()
    sma50 = c.rolling(50, min_periods=50).mean()
    rets = c.pct_change()
    vol20 = rets.rolling(20, min_periods=20).std()
    vol_ref = vol20.rolling(120, min_periods=40).median()

    valid = dist_sma50.notna()
    breadth = (dist_sma50 > 0).where(valid).sum(axis=1) / valid.sum(axis=1).where(valid.sum(axis=1) > 0)
    breadth = breadth.reindex(c.index)

    up = (c > sma50) & (sma20 > sma50) & (breadth >= BREADTH_LINE)
    down = (c < sma50) & (sma20 < sma50) & (breadth < BREADTH_LINE)
    label = pd.Series("NEUTRAL", index=c.index)
    label[up] = "BULLISH"
    label[down] = "BEARISH"

    return pd.DataFrame(
        {
            "regime": label,
            "index_ret_20d": c / c.shift(20) - 1,
            "index_above_sma50": c > sma50,
            "breadth_sma50": breadth,
            "index_vol_20d": vol20,
            "high_vol": (vol20 > HIGH_VOL_MULT * vol_ref).fillna(False),
        }
    )
