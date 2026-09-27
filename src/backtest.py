"""
Walk-forward backtest of the scoring model on committed history.

    uv run src/backtest.py                      # writes reports/backtest.md
    uv run src/backtest.py --out -              # print only

Method (see README "Backtest"):
- Signals at close t, entry at close t+1, exit at close t+1+h (features.forward_labels).
- Returns are reported raw and as *excess* over the same-day equal-weight universe.
- Significance uses one mean per date on non-overlapping dates (every h-th), because
  stocks on the same day and overlapping windows are not independent observations.
- Open/VWAP-based features exist only from daily snapshots onward (history has neither);
  they appear in the feature table once enough dates accumulate.

The removed legacy vote engine's results are archived in reports/legacy_rules_backtest.md.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import classifier as C
import features as F
import history_store
import regime as R
import scoring as S

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "backtest.md"

IC_FEATURES = [
    "ret_5d", "ret_20d", "ret_60d", "dist_sma20", "dist_sma50", "dist_sma120",
    "rsi14", "macd_hist_pct", "range_pos", "drawdown_120", "atr14_pct", "vol_20d",
    "rvol_20", "rturnover_20", "sector_rel_20d",
    # Snapshot-only (need open / VWAP): gap, close vs open, close vs VWAP.
    "gap_1d", "close_vs_open", "close_vs_vwap",
]
# market_rel_20d is ret_20d minus a per-date constant, so its cross-sectional rank IC is identical.
MIN_NAMES_PER_DATE = 30

# Classifier adoption margins over the step-4 score (20-day horizon).
ADOPT_IC_MARGIN = 0.02
ADOPT_EXCESS_MARGIN = 0.0025


def _nonoverlap_t(per_date: pd.Series, h: int) -> float:
    s = per_date.dropna().iloc[::h]
    if len(s) < 3 or s.abs().max() < 1e-12 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s))))


def group_stats(df: pd.DataFrame, mask: pd.Series, h: int, split_date: str) -> dict:
    sub = df[mask & df[f"fwd_ret_{h}"].notna()]
    ex = sub[f"fwd_excess_{h}"]
    dates = sub.index.get_level_values("date")
    per_date = ex.groupby(dates).mean()
    return {
        "n": len(sub),
        "dates": per_date.size,
        "ret": sub[f"fwd_ret_{h}"].mean(),
        "excess": per_date.mean(),
        "median": ex.median(),
        "hit": (ex > 0).mean() if len(ex) else float("nan"),
        "mdd": sub[f"fwd_mdd_{h}"].mean(),
        "t": _nonoverlap_t(per_date, h),
        "excess_h1": per_date[per_date.index < split_date].mean(),
        "excess_h2": per_date[per_date.index >= split_date].mean(),
    }


def feature_ic(df: pd.DataFrame, feature: str, h: int) -> dict:
    """Per-date Spearman IC of feature vs excess return, plus top-minus-bottom quintile spread."""
    sub = df[[feature, f"fwd_excess_{h}"]].dropna()
    ics, spreads = {}, {}
    for d, g in sub.groupby(level="date"):
        if len(g) < MIN_NAMES_PER_DATE or g[feature].nunique() < 5:
            continue
        ics[d] = g[feature].rank().corr(g[f"fwd_excess_{h}"].rank())
        q = pd.qcut(g[feature].rank(method="first"), 5, labels=False)
        spreads[d] = g.loc[q == 4, f"fwd_excess_{h}"].mean() - g.loc[q == 0, f"fwd_excess_{h}"].mean()
    ic = pd.Series(ics, dtype=float)
    sp = pd.Series(spreads, dtype=float)
    return {"ic": ic.mean(), "ic_t": _nonoverlap_t(ic, h), "q5_q1": sp.mean(), "dates": ic.size}


def quintile_table(df: pd.DataFrame, feature: str, h: int) -> pd.Series:
    sub = df[[feature, f"fwd_excess_{h}"]].dropna()
    q = sub.groupby(level="date")[feature].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if len(s) >= MIN_NAMES_PER_DATE else np.nan
    )
    per = sub.assign(q=q).dropna(subset=["q"]).groupby(["q", sub.index.get_level_values("date")])[f"fwd_excess_{h}"]
    return per.mean().groupby(level="q").mean()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(x: float, d: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{d}f}%"


def _num(x: float, d: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _new_listing_note(nl: pd.DataFrame | None) -> str:
    if nl is None or nl.empty or "fwd_ret_20" not in nl:
        return ""
    r = nl["fwd_ret_20"].dropna()
    return (
        f"Those {len(r):,} excluded observations ({nl.index.get_level_values('symbol').nunique()} symbols) had a "
        f"20-day raw return mean of {_pct(r.mean())} but median {_pct(r.median())} — a few runaway listings."
    )


def _stats_table(title: str, rows: list[tuple[str, dict]], h: int) -> list[str]:
    out = [
        f"### {title} — {h}-day horizon",
        "",
        "| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD "
        "| Excess 1st half | Excess 2nd half |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in rows:
        out.append(
            f"| {label} | {s['n']} | {s['dates']} | {_pct(s['ret'])} | {_pct(s['excess'])} | {_num(s['t'])} "
            f"| {_pct(s['median'])} | {_num(s['hit'] * 100, 0)}% | {_pct(s['mdd'], 1)} "
            f"| {_pct(s['excess_h1'])} | {_pct(s['excess_h2'])} |"
        )
    return out + [""]


def build_report(df: pd.DataFrame, horizons: tuple[int, ...] = (5, 20)) -> str:
    dates = sorted(df.index.get_level_values("date").unique())
    labelled = df[df["fwd_ret_1"].notna()]
    split = dates[len(dates) // 2]
    idx_first, idx_last = df.attrs.get("index_first"), df.attrs.get("index_last")

    lines = [
        "# NEPSE scoring backtest",
        "",
        f"History {dates[0]} → {dates[-1]} ({len(dates)} trading days, "
        f"{df.index.get_level_values('symbol').nunique()} symbols, {len(labelled):,} labelled observations). "
        f"NEPSE index over the window: {idx_first} → {idx_last}. Halves split at {split}.",
        "",
        "**Read this first**",
        "",
        "- Entry = close of the day after the signal, exit = close h trading days later. "
        "No entry when the next day had no trades or was locked at the upper limit.",
        "- *Excess* = return minus the same-day equal-weight average of every enterable stock. "
        "It is the number that matters; raw returns mostly reflect the market's direction.",
        "- *t* uses one average per date on every h-th date (non-overlapping). |t| < 2 is "
        "indistinguishable from noise. One year of data is a single market regime.",
        "- `gap_1d`, `close_vs_open` and `close_vs_vwap` need opening prices / VWAP, which exist only "
        "from daily snapshots onward, so they show few or no dates until those accumulate.",
        f"- Universe excludes each new listing's first {F.NEW_LISTING_SESSIONS} sessions "
        "(symbols trading on the first stored day count as seasoned). "
        + _new_listing_note(df.attrs.get("new_listings")),
        "- *Median excess* and *hit rate* (share with excess > 0) matter because returns are skewed: "
        "a mean driven by a few big winners will not show up in a typical trade.",
        f"- Prices are back-adjusted for {df.attrs.get('n_actions', 0)} NOTS bonus/rights/cash-dividend notices "
        "(data/corporate_actions.csv), so those events do not show up as losses.",
        "- Survivorship: only currently listed symbols are in the data.",
        "- The removed legacy vote engine is evaluated in `reports/legacy_rules_backtest.md` (archived).",
        "",
        "## Feature information coefficients",
        "",
        "Spearman rank correlation between each feature and the forward excess return, computed "
        "cross-sectionally per date and averaged. Positive IC = higher value, better future relative "
        f"return. Q5−Q1 = top-quintile minus bottom-quintile excess return. Dates need ≥{MIN_NAMES_PER_DATE} names.",
        "",
    ]
    header = "| Feature | " + " | ".join(f"IC {h}d | t {h}d | Q5−Q1 {h}d" for h in horizons) + " | Dates |"
    lines += [header, "|---|" + "---:|---:|---:|" * len(horizons) + "---:|"]
    ic_rows = []
    for feat in IC_FEATURES:
        res = [feature_ic(df, feat, h) for h in horizons]
        ic_rows.append((feat, res))
    ic_rows.sort(key=lambda fr: -abs(fr[1][-1]["ic"]) if not math.isnan(fr[1][-1]["ic"]) else 0)
    for feat, res in ic_rows:
        cells = " | ".join(f"{_num(r['ic'], 3)} | {_num(r['ic_t'])} | {_pct(r['q5_q1'])}" for r in res)
        lines.append(f"| `{feat}` | {cells} | {res[-1]['dates']} |")
    lines.append("")

    h = horizons[-1]
    lines += [f"## Quintiles — mean {h}-day excess return (Q1 = lowest value)", ""]
    lines += ["| Feature | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---:|---:|---:|---:|---:|"]
    for feat in ("range_pos", "ret_20d", "ret_60d", "dist_sma50", "rsi14", "rvol_20", "sector_rel_20d"):
        qt = quintile_table(df, feat, h)
        lines.append(f"| `{feat}` | " + " | ".join(_pct(qt.get(i, float("nan"))) for i in range(5)) + " |")
    lines.append("")
    return "\n".join(lines)


def score_dataset(df: pd.DataFrame, p: F.Panel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add scoring-model columns and the market regime. Returns (df, weights_by_date, regime_by_date)."""
    calendar = pd.Index(sorted(df.index.get_level_values("date").unique()), name="date")
    scored, weights = S.score_frame(df, calendar)
    reg = R.compute_regime(p.index_close, df["dist_sma50"].unstack()).reindex(calendar)
    scored["regime"] = scored.index.get_level_values("date").map(reg["regime"])
    return scored, weights, reg


def _is_prior(weights: pd.DataFrame) -> pd.Series:
    prior = pd.Series(S.PRIOR_WEIGHTS)
    prior = prior / prior.abs().sum()
    return (weights - prior).abs().max(axis=1) < 1e-12


def _score_ic(df: pd.DataFrame, col: str, h: int) -> dict:
    return feature_ic(df, col, h)


def build_scoring_report(df: pd.DataFrame, weights: pd.DataFrame, reg: pd.DataFrame, h_list=(5, 20)) -> str:
    oos_dates = weights.index[~_is_prior(weights)]
    lines = ["## Scoring model (walk-forward)", ""]
    if len(oos_dates) == 0:
        return "\n".join(lines + ["Not enough realised outcomes yet to learn weights.", ""])
    first = oos_dates[0]
    oos = df[df.index.get_level_values("date") >= first]
    o_dates = sorted(oos.index.get_level_values("date").unique())
    split = o_dates[len(o_dates) // 2]
    everything = pd.Series(True, index=oos.index)

    lines += [
        f"Walk-forward weights start on **{first}**, once {S.MIN_IC_DATES} dates of fully realised "
        f"{S.LABEL_HORIZON}-day outcomes exist. Everything in this section uses only those "
        f"{len(o_dates)} out-of-sample dates (halves split at {split}).",
        "",
        "- `score` = walk-forward weights: each day uses only ICs whose outcome had finished by then.",
        "- `score_prior` = fixed prior weights chosen after looking at the full year in step 3 — "
        "**in-sample**, shown for reference only.",
        "",
        "Weights (signed, sum of |w| = 1):",
        "",
        "| Date | " + " | ".join(S.COMPONENTS) + " |",
        "|---|" + "---:|" * len(S.COMPONENTS),
    ]
    marks = [oos_dates[0]] + list(oos_dates[:: max(1, len(oos_dates) // 4)][1:]) + [oos_dates[-1]]
    prior = pd.Series(S.PRIOR_WEIGHTS) / pd.Series(S.PRIOR_WEIGHTS).abs().sum()
    lines.append("| prior | " + " | ".join(f"{prior[k]:+.2f}" for k in S.COMPONENTS) + " |")
    for d in dict.fromkeys(marks):
        lines.append(f"| {d} | " + " | ".join(f"{weights.loc[d, k]:+.2f}" for k in S.COMPONENTS) + " |")
    lines.append("")

    lines += [
        "### Rank IC of each score vs forward excess return (out-of-sample dates)",
        "",
        "| Score | " + " | ".join(f"IC {h}d | t {h}d | Q5−Q1 {h}d" for h in h_list) + " |",
        "|---|" + "---:|---:|---:|" * len(h_list),
    ]
    for col, label in [("score", "Walk-forward score"), ("score_prior", "Prior score (in-sample)"),
                       ("risk_score", "Risk score (higher = riskier)")]:
        res = [feature_ic(oos, col, h) for h in h_list]
        lines.append(f"| {label} | " + " | ".join(
            f"{_num(r['ic'], 3)} | {_num(r['ic_t'])} | {_pct(r['q5_q1'])}" for r in res) + " |")
    lines.append("")

    for h in h_list:
        rows = [("All observations (baseline)", group_stats(oos, everything, h, split))]
        for c in S.CLASS_ORDER:
            m = oos["classification"] == c
            if m.any():
                rows.append((f"{c}", group_stats(oos, m, h, split)))
        for c in ("STRONG_SETUP", "SETUP"):
            m = oos["classification_prior"] == c
            if m.any():
                rows.append((f"{c} (prior weights, in-sample)", group_stats(oos, m, h, split)))
        lines += _stats_table("Classifications", rows, h)

    counts = reg.loc[o_dates, "regime"].value_counts()
    lines += [
        "### Market regime",
        "",
        "Regime days in the out-of-sample window: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ". "
        "Would skipping STRONG_SETUP names in BEARISH regimes have avoided losses? Judge on **raw** "
        "returns (excess is market-neutral). The live pipeline shows the regime as context but does not "
        "gate on it: the evidence is inconclusive — few BULLISH days, and they were not better than BEARISH ones.",
        "",
    ]
    h = h_list[-1]
    rows = []
    for rg in ("BULLISH", "NEUTRAL", "BEARISH"):
        m = (oos["classification"] == "STRONG_SETUP") & (oos["regime"] == rg)
        if m.any():
            rows.append((f"STRONG_SETUP in {rg}", group_stats(oos, m, h, split)))
    rows.append(("STRONG_SETUP, all regimes", group_stats(oos, oos["classification"] == "STRONG_SETUP", h, split)))
    gated = (oos["classification"] == "STRONG_SETUP") & (oos["regime"] != "BEARISH")
    rows.append(("STRONG_SETUP excluding BEARISH (gate)", group_stats(oos, gated, h, split)))
    lines += _stats_table("Regime gate", rows, h)
    return "\n".join(lines)


def add_classifier_predictions(scored: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    calendar = pd.Index(sorted(scored.index.get_level_values("date").unique()), name="date")
    X = C.design_matrix(scored, reg)
    preds, retrains = C.walk_forward_predict(X, C.target(scored), calendar)
    out = scored.join(preds.add_prefix("p_"))
    dates = out.index.get_level_values("date")
    # Blend: average of the day's percentile of the step-4 score and of the logit probability.
    out["blend"] = (out["score"].groupby(dates).rank(pct=True) + out["p_logit"].groupby(dates).rank(pct=True)) / 2
    return out, retrains


def _top_decile(df: pd.DataFrame, col: str) -> pd.Series:
    pct = df[col].groupby(df.index.get_level_values("date")).rank(pct=True)
    return pct >= 0.90


def build_classifier_report(df: pd.DataFrame, retrains: list[str], h_list=(5, 20)) -> str:
    lines = ["## Learned classifier (walk-forward)", ""]
    common = df[["score", "p_logit", "p_gbm"]].notna().all(axis=1)
    cdf = df[common]
    if cdf.empty:
        return "\n".join(lines + ["Not enough realised outcomes yet to train the classifier.", ""])
    dates = sorted(cdf.index.get_level_values("date").unique())
    split = dates[len(dates) // 2]
    y = C.target(cdf)
    base = y.mean()
    lines += [
        f"Target: P(20-day return beats the same-day universe average). Retrained every {C.RETRAIN_EVERY} "
        f"trading days on outcomes realised before the retrain date; first retrain {retrains[0]}, "
        f"{len(retrains)} retrains. Compared on the {len(dates)} dates where every candidate has an "
        f"out-of-sample value (halves split at {split}). Base rate of the target: {base * 100:.1f}%.",
        "",
        "- `score` — step-4 walk-forward score (the current live ranking).",
        "- `p_logit`, `p_gbm` — classifier probabilities. "
        "`blend` — mean of the day's percentile of `score` and `p_logit`.",
        "",
        "### Ranking quality",
        "",
        "| Candidate | " + " | ".join(f"IC {h}d | t {h}d | Q5−Q1 {h}d" for h in h_list) + " |",
        "|---|" + "---:|---:|---:|" * len(h_list),
    ]
    cands = [("score", "Step-4 score"), ("p_logit", "Logistic"), ("p_gbm", "Gradient boosting"), ("blend", "Blend")]
    for col, label in cands:
        res = [feature_ic(cdf, col, h) for h in h_list]
        lines.append(f"| {label} | " + " | ".join(
            f"{_num(r['ic'], 3)} | {_num(r['ic_t'])} | {_pct(r['q5_q1'])}" for r in res) + " |")
    lines.append("")

    everything = pd.Series(True, index=cdf.index)
    top20 = {}
    for h in h_list:
        rows = [("All observations (baseline)", group_stats(cdf, everything, h, split))]
        for col, label in cands:
            st = group_stats(cdf, _top_decile(cdf, col), h, split)
            rows.append((f"Top 10% by {label}", st))
            if h == h_list[-1]:
                top20[col] = st
        lines += _stats_table("Top decile", rows, h)

    # Adoption rule, fixed in advance: a classifier replaces the score only if it beats it on
    # both 20d IC and 20d top-decile excess return by a margin, in both halves.
    h = h_list[-1]
    ic_score = feature_ic(cdf, "score", h)["ic"]
    verdicts = []
    for col, label in cands[1:3]:
        ic_c = feature_ic(cdf, col, h)["ic"]
        a, b = top20[col], top20["score"]
        wins = (
            ic_c >= ic_score + ADOPT_IC_MARGIN
            and a["excess"] >= b["excess"] + ADOPT_EXCESS_MARGIN
            and a["excess_h1"] > b["excess_h1"]
            and a["excess_h2"] > b["excess_h2"]
        )
        verdicts.append(f"{label}: {'**beats**' if wins else 'does not beat'} the step-4 score")
    lines += [
        f"**Adoption check** (needs +{ADOPT_IC_MARGIN:.2f} IC and +{ADOPT_EXCESS_MARGIN * 100:.2f}pp top-decile "
        f"{h}d excess, and a better top decile in both halves): " + "; ".join(verdicts) + ". "
        "The live digest keeps the step-4 score unless this changes.",
        "",
    ]

    lines += [
        "### Calibration (20-day target)",
        "",
        f"Brier score (lower is better): constant base rate {C.brier(pd.Series(base, index=y.index), y):.4f}, "
        f"logistic {C.brier(cdf['p_logit'], y):.4f}, gradient boosting {C.brier(cdf['p_gbm'], y):.4f}.",
        "",
        "| Decile | Logistic predicted | Logistic realised | GBM predicted | GBM realised |",
        "|---:|---:|---:|---:|---:|",
    ]
    cl, cg = C.calibration_table(cdf["p_logit"], y), C.calibration_table(cdf["p_gbm"], y)
    for b in cl.index:
        lines.append(
            f"| {b + 1} | {cl.loc[b, 'predicted'] * 100:.1f}% | {cl.loc[b, 'realised'] * 100:.1f}% "
            f"| {cg.loc[b, 'predicted'] * 100:.1f}% | {cg.loc[b, 'realised'] * 100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


EVENT_DEDUPE_SESSIONS = 5
EVENT_MAX_LAG_SESSIONS = 5
MIN_EVENTS_REPORTED = 5


def event_rows(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each notice event to the stock's last traded session on or before the notice date
    (notices are posted after the close, so entry is the next close, as for every signal).
    Repeats of the same event type for a symbol within EVENT_DEDUPE_SESSIONS are dropped.
    """
    if events.empty:
        return pd.DataFrame()
    dates = df.index.get_level_values("date")
    day_mean_ret20 = df["ret_20d"].groupby(dates).transform("mean")
    pre20 = (df["ret_20d"] - day_mean_ret20).rename("pre_excess_20")
    cal = pd.Index(sorted(dates.unique()))
    by_sym = {sym: g.index.get_level_values("date") for sym, g in df.groupby(level="symbol")}
    rows, last_seen = [], {}
    for ev in events.sort_values("date").itertuples(index=False):
        sess = by_sym.get(ev.symbol)
        if sess is None:
            continue
        prior = sess[sess <= ev.date]
        if len(prior) == 0:
            continue
        d = prior[-1]
        lag = cal.get_indexer([ev.date])[0] if ev.date in cal else cal.searchsorted(ev.date, side="right") - 1
        if lag - cal.get_loc(d) > EVENT_MAX_LAG_SESSIONS:
            continue
        key = (ev.symbol, ev.event)
        pos = cal.get_loc(d)
        if key in last_seen and pos - last_seen[key] <= EVENT_DEDUPE_SESSIONS:
            continue
        last_seen[key] = pos
        r = df.loc[(d, ev.symbol)]
        rows.append({
            "event": ev.event, "symbol": ev.symbol, "date": d,
            **{f"fwd_excess_{h}": r.get(f"fwd_excess_{h}") for h in F.HORIZONS},
            "fwd_mdd_20": r.get("fwd_mdd_20"), "pre_excess_20": pre20.loc[(d, ev.symbol)],
            "classification": r.get("classification"),
        })
    return pd.DataFrame(rows)


def _event_t(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def build_event_report(ev: pd.DataFrame, window: tuple[str, str]) -> str:
    lines = [
        "## Notice events",
        "",
        f"NEPSE exchange notices typed by `src/events.py`, {window[0]} → {window[1]}, matched to universe stocks. "
        "Signal = the stock's last session on or before the notice (posted after the close); entry at the "
        "next close. *Pre 20d* = excess return over the 20 sessions **before** the event (already happened; "
        "not tradeable). *t events* treats each event as independent; *t dates* averages events on the same "
        "day first, which is the fairer test when events cluster (e.g. bonus season). "
        f"Types with fewer than {MIN_EVENTS_REPORTED} events are listed "
        "but not analysed. Prices are adjusted, so ex-dates are not counted as losses.",
        "",
    ]
    if ev.empty:
        return "\n".join(lines + ["No events matched the price history.", ""])
    lines += [
        "| Event | n | Dates | Pre 20d | After 1d | After 5d | After 10d | After 20d | t events | t dates "
        "| Median 20d | Hit 20d | Avg max DD 20d | Model setup at signal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    small = []
    for etype, g in sorted(ev.groupby("event"), key=lambda kv: -len(kv[1])):
        if len(g) < MIN_EVENTS_REPORTED:
            small.append(f"{etype} ({len(g)})")
            continue
        e20 = g["fwd_excess_20"].dropna()
        by_date = g.dropna(subset=["fwd_excess_20"]).groupby("date")["fwd_excess_20"].mean()
        setup = g["classification"].isin(["STRONG_SETUP", "SETUP"]).mean()
        lines.append(
            f"| `{etype}` | {len(g)} | {g['date'].nunique()} | {_pct(g['pre_excess_20'].mean())} "
            f"| {_pct(g['fwd_excess_1'].mean())} "
            f"| {_pct(g['fwd_excess_5'].mean())} | {_pct(g['fwd_excess_10'].mean())} | {_pct(e20.mean())} "
            f"| {_num(_event_t(e20))} | {_num(_event_t(by_date))} | {_pct(e20.median())} "
            f"| {_num((e20 > 0).mean() * 100, 0)}% "
            f"| {_pct(g['fwd_mdd_20'].mean(), 1)} | {setup * 100:.0f}% |"
        )
    lines.append("")
    if small:
        lines += [f"Too few to analyse: {', '.join(small)}.", ""]
    return "\n".join(lines)


def run(data_dir: Path = history_store.DATA_DIR) -> str:
    p = F.load_panel(data_dir)
    df = F.build_dataset(p)
    df.attrs["index_first"] = f"{p.index_close.dropna().iloc[0]:.0f}"
    df.attrs["index_last"] = f"{p.index_close.dropna().iloc[-1]:.0f}"
    df.attrs["n_actions"] = len(history_store.load_corporate_actions(data_dir))
    scored, weights, reg = score_dataset(df, p)
    with_clf, retrains = add_classifier_predictions(scored, reg)
    events = pd.DataFrame(history_store.load_events(data_dir))
    dates = sorted(df.index.get_level_values("date").unique())
    ev = event_rows(scored, events) if not events.empty else pd.DataFrame()
    return "\n".join([build_report(df), build_scoring_report(scored, weights, reg),
                      build_classifier_report(with_clf, retrains), build_event_report(ev, (dates[0], dates[-1]))])


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the NEPSE rule engine on data/.")
    ap.add_argument("--out", default=str(REPORT_PATH), help="report path, or - for stdout only")
    args = ap.parse_args()
    report = run()
    if args.out == "-":
        print(report)
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
