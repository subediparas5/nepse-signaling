# NEPSE scoring backtest

History 2025-09-24 → 2026-09-24 (229 trading days, 238 symbols, 51,177 labelled observations). NEPSE index over the window: 2654 → 2630. Halves split at 2026-04-07.

**Read this first**

- Entry = close of the day after the signal, exit = close h trading days later. No entry when the next day had no trades or was locked at the upper limit.
- *Excess* = return minus the same-day equal-weight average of every enterable stock. It is the number that matters; raw returns mostly reflect the market's direction.
- *t* uses one average per date on every h-th date (non-overlapping). |t| < 2 is indistinguishable from noise. One year of data is a single market regime.
- `gap_1d`, `close_vs_open` and `close_vs_vwap` need opening prices / VWAP, which exist only from daily snapshots onward, so they show few or no dates until those accumulate.
- Universe excludes each new listing's first 120 sessions (symbols trading on the first stored day count as seasoned). Those 2,419 excluded observations (29 symbols) had a 20-day raw return mean of +13.22% but median -4.29% — a few runaway listings.
- *Median excess* and *hit rate* (share with excess > 0) matter because returns are skewed: a mean driven by a few big winners will not show up in a typical trade.
- Prices are back-adjusted for 470 NOTS bonus/rights/cash-dividend notices (data/corporate_actions.csv), so those events do not show up as losses.
- Survivorship: only currently listed symbols are in the data.
- The removed legacy vote engine is evaluated in `reports/legacy_rules_backtest.md` (archived).

## Feature information coefficients

Spearman rank correlation between each feature and the forward excess return, computed cross-sectionally per date and averaged. Positive IC = higher value, better future relative return. Q5−Q1 = top-quintile minus bottom-quintile excess return. Dates need ≥30 names.

| Feature | IC 5d | t 5d | Q5−Q1 5d | IC 20d | t 20d | Q5−Q1 20d | Dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| `drawdown_120` | 0.098 | 2.46 | +0.61% | 0.240 | 1.91 | +3.19% | 149 |
| `dist_sma120` | 0.107 | 1.72 | +0.67% | 0.234 | 1.33 | +3.06% | 89 |
| `vol_20d` | -0.138 | -3.72 | -0.76% | -0.232 | -3.75 | -2.50% | 188 |
| `atr14_pct` | -0.139 | -4.57 | -0.66% | -0.231 | -4.22 | -2.42% | 194 |
| `range_pos` | 0.081 | 1.80 | +0.47% | 0.203 | 2.45 | +2.42% | 89 |
| `ret_60d` | 0.046 | 1.39 | +0.31% | 0.101 | 0.50 | +1.55% | 148 |
| `dist_sma50` | 0.030 | 0.85 | +0.23% | 0.097 | 1.60 | +1.51% | 159 |
| `ret_20d` | 0.011 | 0.35 | +0.17% | 0.044 | 0.30 | +0.82% | 188 |
| `sector_rel_20d` | 0.008 | 0.10 | +0.14% | 0.042 | -0.03 | +0.87% | 188 |
| `rsi14` | -0.013 | -0.66 | -0.02% | 0.040 | 1.36 | +0.90% | 194 |
| `macd_hist_pct` | -0.050 | -1.74 | -0.30% | -0.038 | -0.39 | -0.35% | 175 |
| `dist_sma20` | -0.023 | -0.97 | -0.07% | 0.035 | 0.34 | +0.65% | 189 |
| `rturnover_20` | -0.053 | -3.54 | -0.41% | -0.017 | -0.76 | -0.08% | 188 |
| `ret_5d` | -0.057 | -1.66 | -0.35% | -0.011 | -0.87 | +0.01% | 203 |
| `rvol_20` | -0.044 | -2.90 | -0.34% | -0.003 | -0.29 | +0.02% | 188 |
| `gap_1d` | — | — | — | — | — | — | 0 |
| `close_vs_open` | — | — | — | — | — | — | 0 |
| `close_vs_vwap` | — | — | — | — | — | — | 0 |

## Quintiles — mean 20-day excess return (Q1 = lowest value)

| Feature | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| `range_pos` | -1.72% | -0.55% | +0.66% | +0.94% | +0.70% |
| `ret_20d` | -0.82% | +0.01% | +0.38% | +0.44% | -0.00% |
| `ret_60d` | -1.32% | -0.03% | +0.46% | +0.68% | +0.23% |
| `dist_sma50` | -1.21% | -0.01% | +0.18% | +0.76% | +0.30% |
| `rsi14` | -0.64% | -0.04% | +0.12% | +0.31% | +0.26% |
| `rvol_20` | -0.42% | +0.10% | +0.41% | +0.29% | -0.40% |
| `sector_rel_20d` | -0.93% | +0.09% | +0.39% | +0.53% | -0.06% |

## Scoring model (walk-forward)

Walk-forward weights start on **2026-01-07**, once 40 dates of fully realised 20-day outcomes exist. Everything in this section uses only those 168 out-of-sample dates (halves split at 2026-05-22).

- `score` = walk-forward weights: each day uses only ICs whose outcome had finished by then.
- `score_prior` = fixed prior weights chosen after looking at the full year in step 3 — **in-sample**, shown for reference only.

Weights (signed, sum of |w| = 1):

| Date | trend | momentum | low_risk | calm | liquidity |
|---|---:|---:|---:|---:|---:|
| prior | +0.30 | +0.10 | +0.30 | +0.20 | +0.10 |
| 2026-01-07 | +0.33 | +0.11 | +0.33 | +0.11 | +0.11 |
| 2026-03-23 | +0.44 | -0.10 | +0.21 | +0.13 | +0.13 |
| 2026-05-22 | +0.23 | -0.00 | +0.37 | +0.20 | +0.20 |
| 2026-07-23 | +0.24 | +0.05 | +0.39 | +0.18 | +0.15 |
| 2026-09-24 | +0.24 | +0.09 | +0.33 | +0.15 | +0.18 |

### Rank IC of each score vs forward excess return (out-of-sample dates)

| Score | IC 5d | t 5d | Q5−Q1 5d | IC 20d | t 20d | Q5−Q1 20d |
|---|---:|---:|---:|---:|---:|---:|
| Walk-forward score | 0.141 | 3.57 | +0.89% | 0.317 | 2.44 | +3.93% |
| Prior score (in-sample) | 0.145 | 3.56 | +0.93% | 0.315 | 2.47 | +3.91% |
| Risk score (higher = riskier) | -0.151 | -3.14 | -0.73% | -0.282 | -3.07 | -2.87% |

### Classifications — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 36693 | 162 | -0.23% | +0.00% | — | -0.06% | 49% | -3.5% | +0.00% | +0.00% |
| STRONG_SETUP | 3716 | 162 | +0.25% | +0.48% | 2.37 | +0.47% | 61% | -2.2% | +0.31% | +0.67% |
| SETUP | 5275 | 162 | +0.06% | +0.28% | 2.07 | +0.18% | 55% | -2.8% | +0.13% | +0.44% |
| WATCH | 5227 | 162 | -0.03% | +0.19% | 1.49 | +0.05% | 51% | -3.1% | +0.16% | +0.23% |
| NEUTRAL | 13166 | 162 | -0.22% | -0.01% | 0.21 | -0.17% | 47% | -3.6% | +0.02% | -0.03% |
| AVOID | 6347 | 162 | -0.62% | -0.34% | -2.45 | -0.38% | 44% | -4.2% | -0.25% | -0.45% |
| HIGH_RISK | 2962 | 162 | -0.86% | -0.67% | -1.82 | -0.87% | 39% | -5.5% | -0.48% | -0.88% |
| STRONG_SETUP (prior weights, in-sample) | 3662 | 162 | +0.26% | +0.49% | 2.31 | +0.48% | 61% | -2.2% | +0.33% | +0.67% |
| SETUP (prior weights, in-sample) | 5253 | 162 | +0.09% | +0.31% | 2.37 | +0.20% | 55% | -2.7% | +0.21% | +0.42% |

### Classifications — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 33201 | 147 | -1.79% | +0.00% | — | +0.05% | 51% | -7.1% | -0.00% | +0.00% |
| STRONG_SETUP | 3360 | 147 | +0.38% | +2.19% | 1.47 | +2.04% | 75% | -4.3% | +1.55% | +3.04% |
| SETUP | 4781 | 147 | -0.35% | +1.45% | 2.04 | +1.22% | 65% | -5.3% | +1.10% | +1.91% |
| WATCH | 4748 | 147 | -0.92% | +0.87% | 1.84 | +0.62% | 59% | -6.1% | +0.69% | +1.12% |
| NEUTRAL | 11956 | 147 | -2.09% | -0.31% | -0.56 | -0.37% | 45% | -7.3% | -0.15% | -0.53% |
| AVOID | 5673 | 147 | -3.38% | -1.51% | -1.61 | -1.37% | 36% | -8.8% | -1.37% | -1.71% |
| HIGH_RISK | 2683 | 147 | -3.94% | -2.31% | -2.70 | -2.43% | 33% | -10.7% | -1.68% | -3.15% |
| STRONG_SETUP (prior weights, in-sample) | 3314 | 147 | +0.35% | +2.16% | 1.49 | +2.08% | 76% | -4.3% | +1.53% | +2.99% |
| SETUP (prior weights, in-sample) | 4770 | 147 | -0.37% | +1.43% | 1.20 | +1.17% | 65% | -5.2% | +1.08% | +1.89% |

### Market regime

Regime days in the out-of-sample window: BEARISH 75, NEUTRAL 56, BULLISH 37. Would skipping STRONG_SETUP names in BEARISH regimes have avoided losses? Judge on **raw** returns (excess is market-neutral). The live pipeline shows the regime as context but does not gate on it: the evidence is inconclusive — few BULLISH days, and they were not better than BEARISH ones.

### Regime gate — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STRONG_SETUP in BULLISH | 826 | 37 | +0.06% | +2.12% | — | +1.66% | 72% | -6.1% | +2.12% | — |
| STRONG_SETUP in NEUTRAL | 1181 | 52 | +1.56% | +2.08% | 0.64 | +2.22% | 73% | -3.3% | +1.37% | +3.84% |
| STRONG_SETUP in BEARISH | 1353 | 58 | -0.46% | +2.32% | 3.08 | +2.18% | 80% | -4.0% | +0.07% | +2.79% |
| STRONG_SETUP, all regimes | 3360 | 147 | +0.38% | +2.19% | 1.47 | +2.04% | 75% | -4.3% | +1.55% | +3.04% |
| STRONG_SETUP excluding BEARISH (gate) | 2007 | 89 | +0.94% | +2.10% | 1.02 | +1.94% | 72% | -4.5% | +1.75% | +3.84% |

## Learned classifier (walk-forward)

Target: P(20-day return beats the same-day universe average). Retrained every 20 trading days on outcomes realised before the retrain date; first retrain 2026-02-09, 8 retrains. Compared on the 148 dates where every candidate has an out-of-sample value (halves split at 2026-06-09). Base rate of the target: 52.1%.

- `score` — step-4 walk-forward score (the current live ranking).
- `p_logit`, `p_gbm` — classifier probabilities. `blend` — mean of the day's percentile of `score` and `p_logit`.

### Ranking quality

| Candidate | IC 5d | t 5d | Q5−Q1 5d | IC 20d | t 20d | Q5−Q1 20d |
|---|---:|---:|---:|---:|---:|---:|
| Step-4 score | 0.162 | 3.85 | +1.04% | 0.341 | 3.46 | +4.21% |
| Logistic | 0.165 | 3.84 | +0.96% | 0.298 | 4.01 | +3.54% |
| Gradient boosting | 0.145 | 4.13 | +0.95% | 0.256 | 2.60 | +3.24% |
| Blend | 0.178 | 4.11 | +1.09% | 0.346 | 3.86 | +4.25% |

### Top decile — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 32253 | 142 | -0.36% | +0.00% | — | +0.01% | 50% | -3.7% | +0.00% | +0.00% |
| Top 10% by Step-4 score | 3337 | 142 | +0.23% | +0.59% | 2.78 | +0.59% | 64% | -2.2% | +0.41% | +0.78% |
| Top 10% by Logistic | 3337 | 142 | +0.15% | +0.50% | 2.30 | +0.50% | 63% | -2.3% | +0.24% | +0.79% |
| Top 10% by Gradient boosting | 3333 | 142 | +0.17% | +0.53% | 2.78 | +0.49% | 63% | -2.5% | +0.51% | +0.54% |
| Top 10% by Blend | 3328 | 142 | +0.20% | +0.56% | 2.96 | +0.56% | 64% | -2.1% | +0.35% | +0.79% |

### Top decile — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 28761 | 127 | -2.30% | +0.00% | — | +0.18% | 52% | -7.4% | -0.00% | +0.00% |
| Top 10% by Step-4 score | 2977 | 127 | +0.10% | +2.38% | 2.84 | +2.25% | 79% | -4.2% | +1.72% | +3.31% |
| Top 10% by Logistic | 2977 | 127 | -0.39% | +1.89% | 2.74 | +2.08% | 75% | -4.8% | +0.90% | +3.26% |
| Top 10% by Gradient boosting | 2973 | 127 | -0.43% | +1.85% | 2.54 | +1.92% | 74% | -5.1% | +1.18% | +2.79% |
| Top 10% by Blend | 2969 | 127 | -0.05% | +2.23% | 3.11 | +2.23% | 78% | -4.3% | +1.43% | +3.35% |

**Adoption check** (needs +0.02 IC and +0.25pp top-decile 20d excess, and a better top decile in both halves): Logistic: does not beat the step-4 score; Gradient boosting: does not beat the step-4 score. The live digest keeps the step-4 score unless this changes.

### Calibration (20-day target)

Brier score (lower is better): constant base rate 0.2495, logistic 0.2393, gradient boosting 0.2445.

| Decile | Logistic predicted | Logistic realised | GBM predicted | GBM realised |
|---:|---:|---:|---:|---:|
| 1 | 21.7% | 38.3% | 24.3% | 35.8% |
| 2 | 28.4% | 35.2% | 31.7% | 37.9% |
| 3 | 33.2% | 40.4% | 35.9% | 47.5% |
| 4 | 37.7% | 44.4% | 39.3% | 51.8% |
| 5 | 42.0% | 49.0% | 42.3% | 48.8% |
| 6 | 46.5% | 53.1% | 45.3% | 51.0% |
| 7 | 51.2% | 56.8% | 48.0% | 52.4% |
| 8 | 56.3% | 60.7% | 50.6% | 56.2% |
| 9 | 62.1% | 64.1% | 54.6% | 63.3% |
| 10 | 70.5% | 79.4% | 65.3% | 76.8% |

## Notice events

NEPSE exchange notices typed by `src/events.py`, 2025-09-24 → 2026-09-24, matched to universe stocks. Signal = the stock's last session on or before the notice (posted after the close); entry at the next close. *Pre 20d* = excess return over the 20 sessions **before** the event (already happened; not tradeable). *t events* treats each event as independent; *t dates* averages events on the same day first, which is the fairer test when events cluster (e.g. bonus season). Types with fewer than 5 events are listed but not analysed. Prices are adjusted, so ex-dates are not counted as losses.

| Event | n | Dates | Pre 20d | After 1d | After 5d | After 10d | After 20d | t events | t dates | Median 20d | Hit 20d | Avg max DD 20d | Model setup at signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `price_adjustment` | 76 | 37 | +1.84% | +0.38% | -0.46% | -1.11% | -2.32% | -3.70 | -1.47 | -2.92% | 31% | -4.7% | 41% |
| `bonus_listing` | 70 | 38 | -0.32% | -0.18% | +0.13% | +0.00% | +0.39% | 0.61 | 0.66 | +0.43% | 53% | -7.2% | 43% |
| `right_listing` | 12 | 9 | +1.77% | -1.69% | -1.36% | -2.94% | -1.69% | -1.00 | -1.27 | -1.09% | 42% | -7.9% | 17% |
| `promoter_conversion` | 8 | 7 | -0.23% | -0.17% | -0.91% | -2.13% | -3.01% | -1.45 | -1.53 | -0.12% | 50% | -9.6% | 25% |

Too few to analyse: merger (2), trading_halt (2), ipo_listing (1), other (1).

