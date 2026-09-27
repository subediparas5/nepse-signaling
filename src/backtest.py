"""
Replay the rule engine over committed history and measure what happened next.

    uv run src/backtest.py                      # writes reports/backtest.md
    uv run src/backtest.py --out -              # print only

Method (see README "Backtest"):
- Signals at close t, entry at close t+1, exit at close t+1+h (features.forward_labels).
- Returns are reported raw and as *excess* over the same-day equal-weight universe.
- Significance uses one mean per date on non-overlapping dates (every h-th), because
  stocks on the same day and overlapping windows are not independent observations.
- Only votes reconstructable from history are replayed; open/VWAP-based votes stay
  neutral until daily snapshots accumulate (history has no open or VWAP).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import features as F
import history_store
from nepse_signal_rules import _liquidity_vote, _sector_relative_vote, _week52_vote, classify_nepse_signal

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "backtest.md"

# Replayable votes: name -> fn(row) -> (buy, sell, reason)
VOTES: dict[str, Callable[[pd.Series], tuple[int, int, str | None]]] = {
    "week52": lambda r: _week52_vote(r["close"], r["range_hi"], r["range_lo"], r["turnover"]),
    "liquidity": lambda r: _liquidity_vote(r["turnover"], r["trades"]),
    "sector_rel": lambda r: _sector_relative_vote(r["diff_pct"], r["sector_median_diff"]),
}

IC_FEATURES = [
    "ret_5d", "ret_20d", "ret_60d", "dist_sma20", "dist_sma50", "dist_sma120",
    "rsi14", "macd_hist_pct", "range_pos", "drawdown_120", "atr14_pct", "vol_20d",
    "rvol_20", "rturnover_20", "sector_rel_20d",
]
# market_rel_20d is ret_20d minus a per-date constant, so its cross-sectional rank IC is identical.
MIN_NAMES_PER_DATE = 30


def build_dataset(p: F.Panel) -> pd.DataFrame:
    """One row per (date, symbol) on days the symbol traded: features, raw fields, labels."""
    feats = F.compute_features(p)
    labels = F.forward_labels(p)
    raw = {
        "close": p.close, "high": p.high, "low": p.low, "open": p.open, "vwap": p.vwap,
        "volume": p.volume, "turnover": p.turnover, "trades": p.trades, "traded": p.traded,
    }
    df = F.to_long({**raw, **feats, **labels})
    df = df[df["traded"].astype(bool)]
    df.attrs["new_listings"] = df[df["new_listing"].astype(bool)]
    df = F.add_excess(df[~df["new_listing"].astype(bool)])
    df["sector"] = df.index.get_level_values("symbol").map(p.sector)
    df["diff_pct"] = df["ret_1d"] * 100
    # Live code compares against the sector median of the same day's movers.
    df["sector_median_diff"] = df.groupby([df.index.get_level_values("date"), "sector"])["diff_pct"].transform(
        "median"
    )
    return df


def replay_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-vote (buy, sell) columns and the rule verdict using only reconstructable fields."""
    out = df.copy()
    for name, fn in VOTES.items():
        res = [fn(r) for _, r in out.iterrows()]
        out[f"vote_{name}"] = [f"buy{b}" if b else (f"sell{s}" if s else "none") for b, s, _ in res]

    verdicts, buys, sells = [], [], []
    for (_, _sym), r in out.iterrows():
        stock = {
            "ltp": r["close"], "close": r["close"], "open": _nan_none(r["open"]), "vwap": _nan_none(r["vwap"]),
            "prev_close": r["close"] / (1 + r["ret_1d"]) if pd.notna(r["ret_1d"]) else None,
            "high": r["high"], "low": r["low"], "turnover": r["turnover"], "transactions": r["trades"],
            "week_52_high": _nan_none(r["range_hi"]), "week_52_low": _nan_none(r["range_lo"]),
            "diff_pct": _nan_none(r["diff_pct"]), "_sector_median_diff": _nan_none(r["sector_median_diff"]),
        }
        if stock["prev_close"] and stock["open"] is not None:
            stock["range_pct"] = (r["high"] - r["low"]) / stock["prev_close"] * 100
        res = classify_nepse_signal(stock, r["sector"] or "")
        verdicts.append(res["signal_verdict"])
        buys.append(res["signal_buy_score"])
        sells.append(res["signal_sell_score"])
    out["verdict"], out["buy_score"], out["sell_score"] = verdicts, buys, sells
    return out


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


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
    everything = pd.Series(True, index=df.index)

    lines = [
        "# NEPSE rule backtest",
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
        "- Replayed votes: 52-week position (trailing ≤240-day range, ≥120 days required), liquidity, "
        "sector-relative day move. Gap, VWAP, open-vs-close and range votes need opens, which "
        "history lacks, so they are neutral here — replayed verdicts are **not** identical to live ones.",
        f"- Universe excludes each new listing's first {F.NEW_LISTING_SESSIONS} sessions "
        "(symbols trading on the first stored day count as seasoned). "
        + _new_listing_note(df.attrs.get("new_listings")),
        "- *Median excess* and *hit rate* (share with excess > 0) matter because returns are skewed: "
        "a mean driven by a few big winners will not show up in a typical trade.",
        "- Survivorship: only currently listed symbols are in the data.",
        "",
        "## Individual rule votes",
        "",
    ]
    for h in horizons:
        rows = [("All observations (baseline)", group_stats(df, everything, h, split))]
        for name in VOTES:
            col = f"vote_{name}"
            for val in sorted(df[col].unique()):
                rows.append((f"`{name}` {val}", group_stats(df, df[col] == val, h, split)))
        lines += _stats_table("Votes", rows, h)

    lines += ["## Replayed verdicts", ""]
    for h in horizons:
        rows = [("All observations (baseline)", group_stats(df, everything, h, split))]
        for v in ["BUY", "LEAN_BUY", "HOLD", "LEAN_SELL", "SELL"]:
            if (df["verdict"] == v).any():
                rows.append((v, group_stats(df, df["verdict"] == v, h, split)))
        lines += _stats_table("Verdicts", rows, h)

    share = df["verdict"].value_counts(normalize=True)
    lines += [
        "Verdict mix over all replayed observations: "
        + ", ".join(f"{k} {v * 100:.0f}%" for k, v in share.items()),
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


def run(data_dir: Path = history_store.DATA_DIR) -> str:
    p = F.load_panel(data_dir)
    df = replay_rules(build_dataset(p))
    df.attrs["index_first"] = f"{p.index_close.dropna().iloc[0]:.0f}"
    df.attrs["index_last"] = f"{p.index_close.dropna().iloc[-1]:.0f}"
    return build_report(df)


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
