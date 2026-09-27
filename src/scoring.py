"""
Evidence-based cross-sectional scoring (replaces the legacy vote engine for ranking).

Each component is a per-date percentile rank in [0, 1] within the scored universe, oriented
so higher is expected to be better (see reports/backtest.md for the evidence):

  trend      price vs SMA50/SMA120, drawdown from 120d high, position in trailing range
  momentum   20d and 60d returns (weak evidence, low prior weight)
  low_risk   inverse of 20d volatility and ATR%        — calmer names outperformed
  calm       inverse of |today's move vs sector|       — one-day shocks either way reversed
  liquidity  20d median turnover

`risk_score` (0-100, higher = riskier) is reported separately from the opportunity side.

Market regime (regime.py) is context only: in the backtest, gating buys on it did not help.

Weights: PRIOR_WEIGHTS until enough outcomes exist, then walk-forward — the mean per-date
rank IC of each component against 20d excess return, using only outcomes that had fully
played out before the scoring date. Live scoring uses the same function, so weights update
as history accumulates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import history_store
import regime as R

COMPONENTS: dict[str, list[tuple[str, int]]] = {
    "trend": [("dist_sma50", 1), ("dist_sma120", 1), ("drawdown_120", 1), ("range_pos", 1)],
    "momentum": [("ret_20d", 1), ("ret_60d", 1)],
    "low_risk": [("vol_20d", -1), ("atr14_pct", -1)],
    "calm": [("abs_sector_rel_1d", -1)],
    "liquidity": [("turnover_med_20", 1)],
}
OPPORTUNITY = ("trend", "momentum", "calm", "liquidity")

# Chosen after the step-3 full-year look, so any backtest of these is in-sample.
PRIOR_WEIGHTS = {"trend": 0.30, "momentum": 0.10, "low_risk": 0.30, "calm": 0.20, "liquidity": 0.10}

LABEL_HORIZON = 20
MIN_IC_DATES = 40
MIN_COMPONENTS = 3
MIN_NAMES = 30

ILLIQUID_TURNOVER = 500_000  # Rs, 20d median
CLASS_ORDER = ["STRONG_SETUP", "SETUP", "WATCH", "NEUTRAL", "AVOID", "HIGH_RISK", "INSUFFICIENT_DATA"]


def _pct_rank(s: pd.Series, dates: pd.Index) -> pd.Series:
    return s.groupby(dates).rank(pct=True)


def component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile components for a long (date, symbol) frame of features."""
    dates = df.index.get_level_values("date")
    src = df.assign(abs_sector_rel_1d=df["sector_rel_1d"].abs())
    out = {}
    for name, feats in COMPONENTS.items():
        parts = []
        for feat, sign in feats:
            r = _pct_rank(src[feat], dates)
            parts.append(r if sign > 0 else 1 - r)
        out[name] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    comp = pd.DataFrame(out, index=df.index)
    comp["n_components"] = comp[list(COMPONENTS)].notna().sum(axis=1)
    return comp


def risk_score(df: pd.DataFrame) -> pd.Series:
    dates = df.index.get_level_values("date")
    parts = [_pct_rank(df[f], dates) for f in ("vol_20d", "atr14_pct")]
    return 100 * pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def component_ic(comp: pd.DataFrame, excess: pd.Series) -> pd.DataFrame:
    """Per-date Spearman IC of each component vs forward excess return (dates with enough names)."""
    dates = comp.index.get_level_values("date")
    ex_rank = excess.groupby(dates).rank()
    rows = {}
    for d, idx in comp.groupby(dates).groups.items():
        e = ex_rank.loc[idx]
        if e.notna().sum() < MIN_NAMES:
            continue
        rows[d] = {k: comp.loc[idx, k].rank().corr(e) for k in COMPONENTS}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def walk_forward_weights(
    ic: pd.DataFrame, calendar: pd.Index, horizon: int = LABEL_HORIZON, min_dates: int = MIN_IC_DATES
) -> pd.DataFrame:
    """
    Weights per scoring date. An IC dated d is usable at t only if its outcome (exit at the
    close of d+1+horizon) is on or before t. Signed, normalised by sum of |w|.
    """
    pos = {d: i for i, d in enumerate(calendar)}
    ic = ic[ic.index.isin(pos.keys())]
    ic_pos = np.array([pos[d] for d in ic.index])
    prior = pd.Series(PRIOR_WEIGHTS)
    rows = {}
    for t in calendar:
        usable = ic[ic_pos + 1 + horizon <= pos[t]]
        if len(usable) < min_dates:
            w = prior
        else:
            # A component without enough realised ICs yet (e.g. SMA120 warm-up) keeps its prior.
            w = usable.mean().where(usable.count() >= min_dates, prior)
            if w.abs().sum() == 0:
                w = prior
        rows[t] = w / w.abs().sum()
    return pd.DataFrame.from_dict(rows, orient="index")[list(COMPONENTS)]


def overall_score(comp: pd.DataFrame, weights: pd.DataFrame | pd.Series) -> pd.Series:
    """0-100. A negative weight means the component counts in reverse (1 - rank)."""
    dates = comp.index.get_level_values("date")
    if isinstance(weights, pd.Series):
        w = pd.DataFrame([weights] * len(comp), index=comp.index)
    else:
        w = weights.reindex(dates).set_axis(comp.index)
    num = pd.Series(0.0, index=comp.index)
    den = pd.Series(0.0, index=comp.index)
    for k in COMPONENTS:
        wk = w[k]
        val = comp[k].where(wk >= 0, 1 - comp[k])
        has = val.notna()
        num += (val * wk.abs()).where(has, 0)
        den += wk.abs().where(has, 0)
    return (100 * num / den.replace(0, np.nan)).where(comp["n_components"] >= MIN_COMPONENTS)


def opportunity_score(comp: pd.DataFrame) -> pd.Series:
    """Prior-weighted opportunity side only (no risk), 0-100, for display."""
    w = pd.Series({k: PRIOR_WEIGHTS[k] for k in OPPORTUNITY})
    vals = comp[list(OPPORTUNITY)]
    return 100 * (vals * w).sum(axis=1, min_count=1) / (vals.notna() * w).sum(axis=1).replace(0, np.nan)


def classify(
    score: pd.Series, risk: pd.Series, turnover_med_20: pd.Series, n_components: pd.Series, new_listing: pd.Series
) -> pd.Series:
    """Map the day's score percentile, risk and liquidity to a label."""
    pct = score.groupby(score.index.get_level_values("date")).rank(pct=True)
    out = pd.Series("NEUTRAL", index=score.index)
    out[pct >= 0.60] = "WATCH"
    out[pct >= 0.75] = "SETUP"
    out[pct >= 0.90] = "STRONG_SETUP"
    out[(pct < 0.20) | (turnover_med_20 < ILLIQUID_TURNOVER)] = "AVOID"
    out[risk >= 90] = "HIGH_RISK"
    out[score.isna() | (n_components < MIN_COMPONENTS) | new_listing.astype(bool)] = "INSUFFICIENT_DATA"
    return out


def score_frame(df: pd.DataFrame, calendar: pd.Index, excess_col: str | None = f"fwd_excess_{LABEL_HORIZON}"):
    """
    Add components, risk, walk-forward and prior scores, and classification to `df`.
    Returns (scored_df, weights_by_date). `excess_col` must be excess within the same universe.
    """
    comp = component_scores(df)
    out = df.join(comp)
    out["risk_score"] = risk_score(df)
    out["opportunity_score"] = opportunity_score(comp)
    out["score_prior"] = overall_score(comp, pd.Series(PRIOR_WEIGHTS))
    if excess_col and excess_col in df:
        weights = walk_forward_weights(component_ic(comp, df[excess_col]), calendar)
    else:
        weights = pd.DataFrame([pd.Series(PRIOR_WEIGHTS)] * len(calendar), index=calendar)
    out["score"] = overall_score(comp, weights)
    out["classification"] = classify(
        out["score"], out["risk_score"], out["turnover_med_20"], out["n_components"], out["new_listing"]
    )
    out["classification_prior"] = classify(
        out["score_prior"], out["risk_score"], out["turnover_med_20"], out["n_components"], out["new_listing"]
    )
    return out, weights


def track_record(scored: pd.DataFrame, weights: pd.DataFrame, h: int = LABEL_HORIZON) -> dict[str, dict]:
    """
    How each class actually did over h days, on dates scored with learned (walk-forward)
    weights and whose outcome is known: n, share beating the universe, mean excess return.
    This is the calibrated "confidence" shown in the digest.
    """
    prior = pd.Series(PRIOR_WEIGHTS) / sum(PRIOR_WEIGHTS.values())
    learned = weights.index[(weights - prior[weights.columns]).abs().max(axis=1) > 1e-12]
    ex = scored[f"fwd_excess_{h}"]
    mask = scored.index.get_level_values("date").isin(learned) & ex.notna()
    sub = scored[mask]
    out = {}
    for cls, g in sub.groupby("classification"):
        e = g[f"fwd_excess_{h}"]
        out[cls] = {"n": int(len(e)), "hit": float((e > 0).mean()), "excess": float(e.mean()), "horizon": h}
    return out


def score_latest(data_dir: Path = history_store.DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """
    Score the most recent stored date for the live digest.

    Returns (per-symbol frame for that date, market context). Weights are learned walk-forward
    from all realised outcomes in data/, exactly as in the backtest.
    """
    p = F.load_panel(data_dir)
    df = F.build_dataset(p)
    calendar = pd.Index(sorted(df.index.get_level_values("date").unique()), name="date")
    scored, weights = score_frame(df, calendar)
    reg = R.compute_regime(p.index_close, df["dist_sma50"].unstack()).reindex(calendar)
    last = calendar[-1]
    today = scored.xs(last, level="date")
    context = {
        "date": last,
        **{k: (None if pd.isna(v) else v) for k, v in reg.loc[last].items()},
        "weights": weights.loc[last].to_dict(),
        "track_record": track_record(scored, weights),
        "weights_learned": len(calendar) > 0 and not np.allclose(
            weights.loc[last].values, (pd.Series(PRIOR_WEIGHTS) / sum(PRIOR_WEIGHTS.values())).values
        ),
    }
    return today, context
