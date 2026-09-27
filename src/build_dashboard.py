"""
Build a self-contained HTML dashboard from data/ (same code paths as the backtest).

    uv run src/build_dashboard.py                 # writes reports/dashboard.html
    uv run src/build_dashboard.py --out /tmp/x.html

The page embeds its data as JSON, so it opens offline. It is regenerated on each run and
not committed (the scheduled workflow uploads it as a build artifact instead).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import backtest as B
import features as F
import history_store
import scoring as S

TEMPLATE = _SRC_DIR / "dashboard_template.html"
OUT = _SRC_DIR.parent / "reports" / "dashboard.html"

# Estimated round-trip friction per unit of portfolio turnover (broker commission + SEBON fee,
# per side, roughly 0.4% for retail NEPSE trades). Applied to both strategies alike.
COST_PER_SIDE = 0.004
CLASS_CODE = {c: i for i, c in enumerate(S.CLASS_ORDER)}


def _r(x, d=4):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return None if math.isnan(f) or math.isinf(f) else round(f, d)


HOLD_SESSIONS = 20  # the horizon the model is trained and evaluated on


def _daily_rebalanced(df: pd.DataFrame, mask: pd.Series, dates: list[str], cost: float) -> tuple[list, float]:
    """
    Equal-weight portfolio of `mask` names rebuilt every session: entered at close t+1, held to
    close t+2 (fwd_ret_1), charged `cost` per side on the fraction of the book that changes.
    Returns (growth of 100 per date, mean daily one-way turnover).
    """
    held_prev: set[str] = set()
    level, curve, turnovers = 100.0, [], []
    live = df[mask & df["fwd_ret_1"].notna()]
    picks_by_date = {d: g.index.get_level_values("symbol") for d, g in live.groupby(level="date")}
    ret_by_date = df["fwd_ret_1"]
    for d in dates:
        held = set(picks_by_date.get(d, []))
        if held:
            ret = float(ret_by_date.loc[[(d, s) for s in held]].mean())
            changed = len(held ^ held_prev) / (2 * len(held)) if held_prev else 1.0
            turnovers.append(changed)
            level *= 1 + ret - 2 * cost * changed
        held_prev = held
        curve.append(_r(level, 3))
    return curve, float(np.mean(turnovers)) if turnovers else float("nan")


def _staggered_hold(p: F.Panel, picks: dict[str, list[str]], dates: list[str], hold: int, cost: float) -> list:
    """
    Each signal date's picks get 1/hold of capital, bought at the next close and held `hold`
    sessions; an empty day's slice sits in cash. One round trip (2 x cost) per slice. Daily
    returns use adjusted closes, so bonus/rights days are not losses. Growth of 100 per date.
    """
    cal = list(p.close.index)
    pos = {d: i for i, d in enumerate(cal)}
    rets = (p.close / p.close.shift(1) - 1).fillna(0.0)
    daily = pd.Series(0.0, index=cal)
    for d in dates:
        names = [n for n in picks.get(d, []) if n in rets.columns]
        if not names:
            continue
        i = pos[d]
        entry = i + 1
        window = cal[entry + 1: entry + 1 + hold]
        if not window:
            continue
        slice_ret = rets.loc[window, names].mean(axis=1) / hold
        daily.loc[window] += slice_ret
        daily.loc[cal[entry]] -= cost / hold
        daily.loc[window[-1]] -= cost / hold
    start = pos[dates[0]]
    growth = 100 * (1 + daily.iloc[start:]).cumprod()
    return [_r(growth.get(d), 3) for d in dates]


def build_payload(data_dir: Path = history_store.DATA_DIR) -> dict:
    p = F.load_panel(data_dir)
    df = B.replay_rules(F.build_dataset(p))
    scored, weights, reg = B.score_dataset(df, p)
    calendar = [str(d) for d in weights.index]
    last = calendar[-1]
    learned = [d for d, prior in zip(calendar, B._is_prior(weights)) if not prior]

    # --- today ---------------------------------------------------------------------------------
    runs, dropped = S.signal_history(scored, last)
    today = scored.xs(last, level="date")
    sectors = p.sector
    stocks = []
    for sym, r in today.iterrows():
        run = runs.get(sym, {})
        stocks.append({
            "symbol": sym, "sector": sectors.get(sym), "cls": r["classification"],
            "close": _r(r["close"], 2), "score": _r(r["score"], 1), "risk": _r(r["risk_score"], 1),
            "opp": _r(r["opportunity_score"], 1), "ret20": _r(r["ret_20d"]), "ret60": _r(r["ret_60d"]),
            "sma50": _r(r["dist_sma50"]), "dd120": _r(r["drawdown_120"]), "vol20": _r(r["vol_20d"]),
            "rsi": _r(r["rsi14"], 1), "turn20": _r(r["turnover_med_20"], 0),
            "comp": {k: _r(r[k], 3) for k in S.COMPONENTS},
            "sessions": run.get("sessions"), "since": _r(run.get("since_ret")), "flagged": run.get("flagged_since"),
            "legacy": r.get("verdict"),
        })
    stocks.sort(key=lambda s: -(s["score"] or -1))

    # --- per-stock history -------------------------------------------------------------------
    cls_w = scored["classification"].unstack("symbol").reindex(calendar)
    score_w = scored["score"].unstack("symbol").reindex(calendar)
    close_w = p.close.reindex(calendar)
    sma50_w = p.close.rolling(50, min_periods=50).mean().reindex(calendar)
    history = {}
    for sym in today.index:
        history[sym] = {
            "close": [_r(v, 2) for v in close_w[sym]],
            "sma50": [_r(v, 2) for v in sma50_w[sym]],
            "score": [_r(v, 1) for v in score_w[sym]] if sym in score_w else [],
            "cls": [CLASS_CODE.get(v) if isinstance(v, str) else None for v in cls_w[sym]] if sym in cls_w else [],
        }

    # --- strategies (out-of-sample dates only) -----------------------------------------------
    oos = scored[scored.index.get_level_values("date").isin(learned)]
    strong = oos["classification"] == "STRONG_SETUP"
    legacy = oos["verdict"].isin(["BUY", "LEAN_BUY"])

    def picks(mask: pd.Series) -> dict[str, list[str]]:
        return {d: list(g.index.get_level_values("symbol")) for d, g in oos[mask].groupby(level="date")}

    strong_hold = _staggered_hold(p, picks(strong), learned, HOLD_SESSIONS, COST_PER_SIDE)
    legacy_hold = _staggered_hold(p, picks(legacy), learned, HOLD_SESSIONS, COST_PER_SIDE)
    strong_daily_net, strong_turn = _daily_rebalanced(oos, strong, learned, COST_PER_SIDE)
    strong_daily_gross, _ = _daily_rebalanced(oos, strong, learned, 0.0)
    everyone = oos["close"].notna()
    univ_hold = _staggered_hold(p, picks(everyone), learned, HOLD_SESSIONS, 0.0)
    idx = p.index_close.reindex(calendar)
    idx_curve = [_r(v, 3) for v in 100 * idx.reindex(learned) / idx.reindex(learned).iloc[0]]

    # --- class track record, weights, feature ICs --------------------------------------------
    rec_all = S.track_record(scored, weights)
    rec_recent = S.track_record(scored, weights, last_sessions=S.RECENT_SESSIONS)
    split = learned[len(learned) // 2]
    classes = []
    for c in S.CLASS_ORDER:
        if c == "INSUFFICIENT_DATA":
            continue
        m = oos["classification"] == c
        st20, st5 = B.group_stats(oos, m, 20, split), B.group_stats(oos, m, 5, split)
        classes.append({
            "cls": c, "n": st20["n"], "excess20": _r(st20["excess"]), "hit20": _r(st20["hit"], 3),
            "excess5": _r(st5["excess"]), "mdd20": _r(st20["mdd"]),
            "recent_hit20": _r((rec_recent.get(c) or {}).get("hit"), 3),
            "recent_excess20": _r((rec_recent.get(c) or {}).get("excess")),
        })
    legacy_rows = []
    for v in ("BUY", "LEAN_BUY"):
        m = oos["verdict"] == v
        if m.any():
            st = B.group_stats(oos, m, 20, split)
            legacy_rows.append({"cls": f"Legacy {v}", "n": st["n"], "excess20": _r(st["excess"]),
                                "hit20": _r(st["hit"], 3)})
    ics = []
    for feat in B.IC_FEATURES:
        res = B.feature_ic(scored, feat, 20)
        ics.append({"feature": feat, "ic": _r(res["ic"], 3), "t": _r(res["ic_t"], 2), "q5q1": _r(res["q5_q1"])})
    ics.sort(key=lambda r: -(r["ic"] or 0))
    ic_score = B.feature_ic(oos, "score", 20)
    ic_legacy = B.feature_ic(oos, "legacy_net", 20)

    counts = today["classification"].value_counts().to_dict()
    r_last = reg.loc[last]
    return {
        "as_of": last,
        "first_date": calendar[0],
        "n_symbols": int(today.shape[0]),
        "calendar": calendar,
        "regime": {
            "label": r_last["regime"], "index_ret_20d": _r(r_last["index_ret_20d"]),
            "breadth": _r(r_last["breadth_sma50"], 3), "high_vol": bool(r_last["high_vol"]),
            "index_close": _r(idx.iloc[-1], 2),
        },
        "regime_history": [r if isinstance(r, str) else None for r in reg["regime"].reindex(calendar)],
        "index_close": [_r(v, 2) for v in idx],
        "counts": {k: int(v) for k, v in counts.items()},
        "track_record": {"all": rec_all.get("STRONG_SETUP"), "recent": rec_recent.get("STRONG_SETUP"),
                         "recent_sessions": S.RECENT_SESSIONS},
        "stocks": stocks,
        "dropped": dropped,
        "history": history,
        "strategies": {
            "dates": learned,
            "cost_per_side": COST_PER_SIDE,
            "hold": HOLD_SESSIONS,
            "strong_daily_turnover": _r(strong_turn, 3),
            "series": [
                {"key": "strong_hold", "label": f"Strong setups, {HOLD_SESSIONS}-day hold (net)",
                 "values": strong_hold},
                {"key": "legacy_hold", "label": f"Legacy BUY + LEAN_BUY, {HOLD_SESSIONS}-day hold (net)",
                 "values": legacy_hold},
                {"key": "universe", "label": f"All stocks, {HOLD_SESSIONS}-day hold (no costs)", "values": univ_hold},
                {"key": "index", "label": "NEPSE index", "values": idx_curve},
                {"key": "strong_daily_net", "label": "Strong setups, rebuilt daily (net)", "values": strong_daily_net},
                {"key": "strong_daily_gross", "label": "Strong setups, rebuilt daily (before costs)",
                 "values": strong_daily_gross},
            ],
        },
        "classes": classes,
        "legacy_classes": legacy_rows,
        "weights": {
            "dates": calendar,
            "learned_from": learned[0] if learned else None,
            "series": {k: [_r(v, 3) for v in weights[k]] for k in S.COMPONENTS},
        },
        "ics": ics,
        "score_ic": {"model": _r(ic_score["ic"], 3), "model_t": _r(ic_score["ic_t"], 2),
                     "legacy": _r(ic_legacy["ic"], 3), "legacy_t": _r(ic_legacy["ic_t"], 2)},
        "n_actions": len(history_store.load_corporate_actions(data_dir)),
    }


def render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    data = data.replace("</", "<\\/")  # never close the <script> early
    html = TEMPLATE.read_text(encoding="utf-8")
    marker = "/*__DASHBOARD_DATA__*/null"
    if marker not in html:
        raise RuntimeError(f"{TEMPLATE} is missing the data marker")
    return html.replace(marker, data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the NEPSE dashboard HTML.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(build_payload()), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
