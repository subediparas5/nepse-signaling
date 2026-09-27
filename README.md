# NEPSE Signaling

Automated buy/sell signal engine for the Nepal Stock Exchange (NEPSE). Runs on a schedule, scores listed equities using rule-based technical signals from **official NEPSE NOTS data**, optionally reviewed by DeepSeek (skipped when `OPEN_AI_API_KEY` is unset), and sends alerts to Telegram.

## How It Works

```
www.nepalstock.com.np (NOTS: listing + prices + 52w)  ──▶  Rule Engine  ──▶  DeepSeek  ──▶  Telegram
```

Market data is read through the open-source [`nepse-data-api`](https://pypi.org/project/nepse-data-api/) package, which implements NEPSE’s authenticated WASM token flow. Traffic goes to **nepalstock.com.np**, not aggregator websites.

**Not included:** bulk P/E, P/B, EPS, NPL, promoter %, or dividend yield from NOTS in the paths used here. Fundamental votes in the rule engine stay neutral unless you later attach your own fields to each stock dict.

### Technical signals (from NOTS + live merge)

OHLC, VWAP (average traded price), volume, turnover, transaction count, 52-week high/low, daily change, gaps, sector-relative strength, intraday range. **No** 120d/180d moving averages in this feed — MA-related votes are effectively neutral. **IPO** verdict applies only when `ma120` is present and equal to `0` (e.g. if you enrich data elsewhere).

### Telegram output

- **Group** (`TELEGRAM_CHAT_ID`): one short HTML message — top **5** BUY rows (monospace table), optional **LLM** tickers on one line, and up to **8** near-52w-low rows (`8/total` in the header).

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

### Tests

```bash
uv run pytest
```

### GitHub Actions

Workflow: `.github/workflows/schedule.yml` (cron in **Asia/Kathmandu**).

**Secrets:** `OPEN_AI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Project structure

```
src/
  main_signaling.py     # Orchestration, LLM, Telegram
  nepse_official.py     # NOTS listing + per-symbol market merge
  nepse_signal_rules.py # Scoring and verdicts
.github/workflows/
  schedule.yml
tests/                  # pytest, no network
```

## Scoring

Independent buy/sell scores from weighted votes. Thresholds depend on whether any fundamental field
(P/E, EPS, P/B, dividend, promoter %, ROE, NPL — see `FUNDAMENTAL_FIELDS`) is present on the stock.
The NOTS feed carries none, so the **price-only** column is what runs today.

| Verdict | Price-only | With fundamentals |
|---------|------------|-------------------|
| **BUY** | buy ≥ 4 and buy ≥ sell + 3 | buy ≥ 6 and buy ≥ sell + 3 |
| **SELL** | sell ≥ 4 and sell ≥ buy + 3 | sell ≥ 6 and sell ≥ buy + 3 |
| **LEAN_BUY** | buy ≥ 3 and buy ≥ sell + 1 | buy ≥ 4 and buy ≥ sell + 1 |
| **LEAN_SELL** | sell ≥ 3 and sell ≥ buy + 1 | sell ≥ 4 and sell ≥ buy + 1 |
| **HOLD** | Otherwise | Otherwise |
| **IPO** | `ma120` present and equal to 0 | same |

Confidence: `round(abs(buy - sell) / (buy + sell) * 100)` when the denominator is positive.

## License

For personal use.
