# NEPSE Signaling

Daily ranking engine for the Nepal Stock Exchange (NEPSE). Runs on a schedule, records **official NEPSE NOTS data** into a committed history, ranks listed equities with a backtested cross-sectional model, optionally has DeepSeek annotate the top names (skipped when `OPEN_AI_API_KEY` is unset), and sends a digest to Telegram.

## How It Works

```
NOTS (listing + prices + 52w + index) ─▶ data/ history ─▶ features ─▶ scoring model ─▶ DeepSeek notes ─▶ Telegram
```

Market data is read through the open-source [`nepse-data-api`](https://pypi.org/project/nepse-data-api/) package, which implements NEPSE’s authenticated WASM token flow. Traffic goes to **nepalstock.com.np**, not aggregator websites.

**Not included:** P/E, P/B, EPS, NPL, promoter % or dividend yield — NOTS does not provide them in the paths used here.

### Scoring model (`src/scoring.py`)

Each day, every seasoned stock gets per-date percentile components, oriented so higher is better:

| Component | Built from | Evidence (reports/backtest.md) |
|-----------|-----------|-------------------------------|
| trend | price vs SMA50/SMA120, drawdown from 120d high, trailing-range position | stocks near lows lagged |
| momentum | 20d / 60d return | weak |
| low_risk | inverse 20d volatility and ATR% | strongest single factor |
| calm | inverse of today's move vs sector | one-day shocks either way reversed |
| liquidity | 20d median turnover | modest |

`score` (0-100) is the weighted blend. Weights are **walk-forward**: the mean daily rank correlation of
each component with 20-day excess return, using only outcomes that had fully played out before the
scoring date (prior weights until 40 such dates exist). `risk_score` (volatility rank) is reported
separately. Classes from the day's score percentile:

| Class | Rule |
|-------|------|
| STRONG_SETUP | top 10% |
| SETUP | top 25% |
| WATCH | top 40% |
| NEUTRAL | otherwise |
| AVOID | bottom 20%, or 20d median turnover < Rs 0.5M |
| HIGH_RISK | risk_score ≥ 90 |
| INSUFFICIENT_DATA | new listing (first 120 sessions) or < 3 components |

A walk-forward learned classifier (`src/classifier.py`) is benchmarked against this score in the
backtest with a pre-set adoption rule; it has not beaten it, so the live digest does not use it. The
digest's confidence line is instead each class's realised walk-forward track record.

The market regime (`src/regime.py`: NEPSE vs SMA20/50 plus breadth) is shown as context; the backtest
found no reliable benefit from gating on it. Scores are **relative** — a STRONG_SETUP can still fall in a
falling market.

The original buy/sell vote engine was removed after its backtest showed no edge (its BUY signals trailed
the average stock by about 1.9% over 20 days); the evidence is archived in `reports/legacy_rules_backtest.md`.
Its open/VWAP ideas live on as features (`gap_1d`, `close_vs_open`, `close_vs_vwap`) that the backtest
evaluates once daily snapshots accumulate.

### Telegram output

One HTML message to `TELEGRAM_CHAT_ID`:

- market regime line (NEPSE 20d return, breadth, volatility flag)
- top 8 setups: price, score, risk, 20d return, **sessions flagged** and **return since first flagged**
  (adjusted prices), `*` = strong setup
- track record of strong setups over the last 90 sessions and all history (share that beat the market over 20d)
- optional DeepSeek notes — one summary and up to 3 risks per setup
- names that were a setup last session but no longer are, and HIGH_RISK / AVOID counts

### DeepSeek (optional)

DeepSeek explains the ranking; it cannot change it. It receives each setup's model fields and signal
history and must reply with JSON `{"notes": [{"symbol", "summary", "risks": []}]}`. The reply is
validated: unknown or duplicate symbols, empty summaries and non-JSON output are dropped, text is
length-capped and HTML-escaped. Any failure (no key, API error, bad JSON) sends the digest without notes.
Model: `DEEPSEEK_MODEL` (default `deepseek-reasoner`).

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- DeepSeek API key ([platform.deepseek.com](https://platform.deepseek.com)) — optional; without it the digest omits the LLM section
- Telegram bot token and chat ID

### Local

```bash
uv sync
export OPEN_AI_API_KEY="sk-..."      # optional
export TELEGRAM_BOT_TOKEN="123:ABC..."
export TELEGRAM_CHAT_ID="-100..."   # group: short digest
uv run src/main_signaling.py
```

The first run may take **1–3 minutes** while security detail is fetched for each symbol.

### History (`data/`)

Every run appends to plain CSVs committed to the repo — the research record for backtesting:

| Path | Rows | Source |
|------|------|--------|
| `data/prices/YYYY-MM.csv` | one per (date, symbol): OHLC, VWAP, volume, turnover, trades, 52w | backfill + daily snapshot |
| `data/index/nepse.csv` | one per date: NEPSE index OHLC, turnover, trades | backfill |
| `data/signals/YYYY-MM.csv` | one per (date, symbol): class, score, opportunity, risk, regime, close | daily run |
| `data/securities.csv` | one per symbol: NOTS id, sector, first/last seen | both |
| `data/corporate_actions.csv` | one per (date, symbol): bonus/rights/cash-dividend adjustment factor | NOTS news notices |
| `data/events.csv` | one per (notice, symbol): typed exchange notice (listing, halt, ex-date, …) | NOTS news notices |

Prices are stored **raw**; `features.build_panel` back-adjusts them with `corporate_actions.csv`
(parsed from NOTS "Price Adjusted" notices: factor = adjusted ÷ previous close) so bonus and rights
issues do not look like crashes.

`date` is the NEPSE business date the data describes (the 09:00 NPT run records the previous session).
Writes are upserts and never blank out a stored value, so re-runs are safe and unchanged data produces no diff.

**NOTS only serves ~1 trading year of history**, and its history endpoint has no open, VWAP or 52w
fields — those exist only from daily snapshots onward. Anything older than a year survives only in git.
Only currently listed equities are backfilled; symbols are never removed from `securities.csv` once seen.

```bash
uv run src/backfill_history.py              # full available window (~3 min, one request per symbol)
uv run src/backfill_history.py --days 30    # recent window, fills missed days
```

### Backtest

```bash
uv run src/backtest.py        # replays the rules over data/, writes reports/backtest.md
```

Features (`src/features.py`) are point-in-time: returns, SMA distances, RSI, ATR, MACD, volatility,
relative volume/turnover, trailing range position, drawdown, sector/market-relative strength.
The backtest enters at the **next day's close** (history has no opens), skips limit-up-locked and
no-trade days, reports returns in excess of the same-day universe average, and excludes each new
listing's first 120 sessions. Open/VWAP features exist only from daily snapshots onward.
See the header of `reports/backtest.md` for how to read it.

Notice events (`src/events.py`, keyword rules in English and Nepali) are evaluated as an event study
in the same report: excess return before and after each event type, with a date-clustered t. So far only
the bonus ex-date shows a pattern (run-up before, about -2.3% vs the market over the next 20 sessions),
and it comes from a single bonus season, so it is reported but not used in the score.

### Dashboard

```bash
uv run src/build_dashboard.py     # writes reports/dashboard.html (self-contained, ~1.3 MB)
```

One offline HTML page built from `data/` with the same code as the backtest: today's sortable ranking,
per-stock adjusted price / daily class / score history, growth of Rs 100 for the model vs the average
stock and the NEPSE index (out-of-sample sessions, net of an estimated 0.4% per trade
side), class track record, feature correlations and component weights over time. It is not committed:
each scheduled run publishes it to GitHub Pages at **https://subediparas5.github.io/nepse-signaling/**
(public, like the repo) and also attaches it to the run as the `nepse-dashboard` build artifact.

The strategy comparison uses a staggered 20-session hold (each day's picks get 1/20 of capital), the
horizon the model is evaluated on. Rebuilding the list daily is shown too: at ~33% daily turnover,
costs outweigh the edge.

### Tests

```bash
uv run pytest
```

### GitHub Actions

Workflow: `.github/workflows/schedule.yml`, Monday–Friday (covering the Sunday–Thursday sessions) at
**03:37 NPT with a 06:37 NPT backup** (cron in Asia/Kathmandu). GitHub has started scheduled runs 4–5 h
late, so this keeps the digest ahead of the 11:00 open. Each business date gets one digest: sent dates
are recorded in `data/digests.csv`, and a later run for the same date exits early (this also skips
holidays, when the business date does not change). After the digest it
backfills the last 30 days and commits any `data/` changes back to the branch (`contents: write`).

**Secrets:** `OPEN_AI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Project structure

```
src/
  main_signaling.py     # Orchestration, LLM, Telegram
  nepse_official.py     # NOTS listing + per-symbol market merge
  scoring.py            # Walk-forward scoring model and classes
  regime.py             # NEPSE market regime
  history_store.py      # CSV history store (upserts, deterministic output)
  backfill_history.py   # Pull NOTS price/index history into data/
  features.py           # Point-in-time features + forward-return labels
  backtest.py           # Rule replay, vote/feature statistics, model comparisons, report
  classifier.py         # Walk-forward logistic / gradient-boosting benchmark (not used live)
  events.py             # Types NOTS notices into per-symbol events
  build_dashboard.py     # Dashboard data + HTML render
  dashboard_template.html
reports/backtest.md     # Latest backtest output
reports/legacy_rules_backtest.md  # Archived evidence for removing the old vote engine
data/                   # Committed history (see above)
.github/workflows/
  schedule.yml
tests/                  # pytest, no network
```

## License

For personal use.
