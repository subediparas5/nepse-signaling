# NEPSE rule backtest

History 2025-09-24 → 2026-09-24 (229 trading days, 238 symbols, 51,177 labelled observations). NEPSE index over the window: 2654 → 2630. Halves split at 2026-04-07.

**Read this first**

- Entry = close of the day after the signal, exit = close h trading days later. No entry when the next day had no trades or was locked at the upper limit.
- *Excess* = return minus the same-day equal-weight average of every enterable stock. It is the number that matters; raw returns mostly reflect the market's direction.
- *t* uses one average per date on every h-th date (non-overlapping). |t| < 2 is indistinguishable from noise. One year of data is a single market regime.
- Replayed votes: 52-week position (trailing ≤240-day range, ≥120 days required), liquidity, sector-relative day move. Gap, VWAP, open-vs-close and range votes need opens, which history lacks, so they are neutral here — replayed verdicts are **not** identical to live ones.
- Universe excludes each new listing's first 120 sessions (symbols trading on the first stored day count as seasoned). Those 2,419 excluded observations (29 symbols) had a 20-day raw return mean of +13.22% but median -4.29% — a few runaway listings.
- *Median excess* and *hit rate* (share with excess > 0) matter because returns are skewed: a mean driven by a few big winners will not show up in a typical trade.
- Prices are back-adjusted for 470 NOTS bonus/rights/cash-dividend notices (data/corporate_actions.csv), so those events do not show up as losses.
- Survivorship: only currently listed symbols are in the data.

## Individual rule votes

### Votes — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 50235 | 223 | -0.17% | +0.00% | — | -0.09% | 48% | -3.4% | +0.00% | +0.00% |
| `week52` buy1 | 6004 | 104 | -0.64% | -0.12% | -1.97 | -0.11% | 48% | -3.5% | — | -0.12% |
| `week52` buy2 | 4815 | 104 | -0.40% | -0.35% | -1.30 | -0.22% | 46% | -4.2% | — | -0.35% |
| `week52` none | 38361 | 223 | -0.04% | +0.11% | 4.12 | -0.08% | 48% | -3.3% | +0.00% | +0.22% |
| `week52` sell1 | 618 | 96 | -0.95% | +0.00% | -0.73 | +0.15% | 53% | -3.7% | — | +0.00% |
| `week52` sell2 | 307 | 87 | -1.54% | -1.14% | -1.62 | -0.15% | 46% | -4.5% | — | -1.14% |
| `week52` sell3 | 130 | 104 | -0.42% | +0.43% | 0.77 | +0.85% | 64% | -1.5% | — | +0.43% |
| `liquidity` buy1 | 25690 | 223 | -0.18% | +0.08% | 1.03 | -0.09% | 48% | -3.5% | +0.07% | +0.10% |
| `liquidity` none | 12186 | 223 | -0.12% | +0.01% | -0.40 | -0.06% | 49% | -3.4% | +0.05% | -0.04% |
| `liquidity` sell1 | 12359 | 223 | -0.19% | -0.09% | -0.65 | -0.14% | 47% | -3.3% | -0.14% | -0.03% |
| `sector_rel` buy1 | 4945 | 222 | -0.74% | -0.60% | -3.00 | -0.82% | 38% | -4.5% | -0.52% | -0.69% |
| `sector_rel` none | 40964 | 223 | -0.06% | +0.12% | 5.69 | +0.00% | 50% | -3.2% | +0.11% | +0.13% |
| `sector_rel` sell1 | 4326 | 222 | -0.54% | -0.54% | -5.08 | -0.46% | 43% | -4.4% | -0.45% | -0.64% |

### Votes — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 46743 | 208 | -0.81% | +0.00% | — | -0.14% | 49% | -6.4% | +0.00% | +0.00% |
| `week52` buy1 | 5042 | 89 | -3.24% | -0.59% | -1.70 | -0.24% | 47% | -7.6% | — | -0.59% |
| `week52` buy2 | 3550 | 89 | -3.36% | -1.91% | -2.29 | -1.23% | 35% | -8.9% | — | -1.91% |
| `week52` none | 37232 | 208 | -0.19% | +0.38% | 2.24 | -0.02% | 50% | -6.0% | +0.00% | +0.84% |
| `week52` sell1 | 548 | 82 | -2.45% | +0.82% | -0.13 | +1.40% | 63% | -7.0% | — | +0.82% |
| `week52` sell2 | 259 | 72 | -4.29% | -0.87% | -1.04 | -0.04% | 50% | -8.5% | — | -0.87% |
| `week52` sell3 | 112 | 89 | -1.20% | +2.10% | 1.55 | +1.21% | 65% | -2.6% | — | +2.10% |
| `liquidity` buy1 | 24245 | 208 | -0.39% | +0.46% | 1.26 | +0.12% | 51% | -6.4% | +0.31% | +0.65% |
| `liquidity` none | 11236 | 208 | -0.90% | -0.25% | -0.91 | -0.26% | 47% | -6.3% | -0.15% | -0.37% |
| `liquidity` sell1 | 11262 | 208 | -1.64% | -0.68% | -1.33 | -0.53% | 44% | -6.4% | -0.85% | -0.47% |
| `sector_rel` buy1 | 4592 | 207 | -1.82% | -1.14% | -1.50 | -1.23% | 38% | -8.1% | -0.73% | -1.64% |
| `sector_rel` none | 38200 | 208 | -0.59% | +0.24% | 2.01 | +0.06% | 51% | -6.0% | +0.20% | +0.29% |
| `sector_rel` sell1 | 3951 | 207 | -1.75% | -1.36% | -4.68 | -1.13% | 40% | -8.0% | -0.95% | -1.86% |

## Replayed verdicts

### Verdicts — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 50235 | 223 | -0.17% | +0.00% | — | -0.09% | 48% | -3.4% | +0.00% | +0.00% |
| BUY | 64 | 40 | +0.26% | -0.08% | -1.97 | -0.53% | 45% | -4.3% | — | -0.08% |
| LEAN_BUY | 1376 | 104 | -0.33% | -0.33% | -0.88 | -0.30% | 44% | -4.2% | — | -0.33% |
| HOLD | 48643 | 223 | -0.16% | +0.01% | 1.50 | -0.09% | 48% | -3.4% | +0.00% | +0.02% |
| LEAN_SELL | 46 | 39 | -3.10% | -3.03% | -1.27 | -1.68% | 30% | -7.7% | — | -3.03% |
| SELL | 106 | 104 | +0.22% | +0.78% | 1.19 | +1.03% | 72% | -0.2% | — | +0.78% |

### Verdicts — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 46743 | 208 | -0.81% | +0.00% | — | -0.14% | 49% | -6.4% | +0.00% | +0.00% |
| BUY | 39 | 30 | -3.32% | -1.89% | — | -1.16% | 36% | -8.8% | — | -1.89% |
| LEAN_BUY | 966 | 89 | -2.55% | -1.16% | -1.10 | -0.63% | 42% | -8.4% | — | -1.16% |
| HOLD | 45607 | 208 | -0.77% | +0.02% | 1.06 | -0.12% | 49% | -6.3% | +0.00% | +0.05% |
| LEAN_SELL | 40 | 34 | -9.09% | -6.26% | — | -5.12% | 25% | -13.2% | — | -6.26% |
| SELL | 91 | 89 | +0.53% | +3.12% | 1.56 | +2.26% | 73% | -0.4% | — | +3.12% |

Verdict mix over all replayed observations: HOLD 97%, LEAN_BUY 3%, SELL 0%, BUY 0%, LEAN_SELL 0%

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
- Legacy = the replayed old vote engine on the same dates.

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
| Legacy buy − sell | -0.035 | -0.80 | -0.15% | -0.027 | 0.35 | -0.43% |

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
| Legacy BUY | 64 | 40 | +0.26% | -0.08% | -1.97 | -0.53% | 45% | -4.3% | +1.17% | -0.39% |
| Legacy LEAN_BUY | 1376 | 104 | -0.33% | -0.33% | -0.88 | -0.30% | 44% | -4.2% | -0.95% | -0.12% |

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
| Legacy BUY | 39 | 30 | -3.32% | -1.89% | — | -1.16% | 36% | -8.8% | +0.29% | -2.68% |
| Legacy LEAN_BUY | 966 | 89 | -2.55% | -1.16% | -1.10 | -0.63% | 42% | -8.4% | -0.36% | -1.49% |

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

