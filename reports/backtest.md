# NEPSE rule backtest

History 2025-09-24 → 2026-09-24 (229 trading days, 238 symbols, 51,177 labelled observations). NEPSE index over the window: 2654 → 2630. Halves split at 2026-04-07.

**Read this first**

- Entry = close of the day after the signal, exit = close h trading days later. No entry when the next day had no trades or was locked at the upper limit.
- *Excess* = return minus the same-day equal-weight average of every enterable stock. It is the number that matters; raw returns mostly reflect the market's direction.
- *t* uses one average per date on every h-th date (non-overlapping). |t| < 2 is indistinguishable from noise. One year of data is a single market regime.
- Replayed votes: 52-week position (trailing ≤240-day range, ≥120 days required), liquidity, sector-relative day move. Gap, VWAP, open-vs-close and range votes need opens, which history lacks, so they are neutral here — replayed verdicts are **not** identical to live ones.
- Universe excludes each new listing's first 120 sessions (symbols trading on the first stored day count as seasoned). Those 2,419 excluded observations (29 symbols) had a 20-day raw return mean of +13.21% but median -4.29% — a few runaway listings.
- *Median excess* and *hit rate* (share with excess > 0) matter because returns are skewed: a mean driven by a few big winners will not show up in a typical trade.
- Survivorship: only currently listed symbols are in the data.

## Individual rule votes

### Votes — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 50235 | 223 | -0.23% | -0.00% | — | -0.05% | 49% | -3.5% | -0.00% | +0.00% |
| `week52` buy1 | 6775 | 104 | -0.65% | -0.05% | -1.14 | -0.00% | 50% | -3.4% | — | -0.05% |
| `week52` buy2 | 5748 | 104 | -0.34% | -0.28% | -1.56 | -0.13% | 48% | -4.0% | — | -0.28% |
| `week52` none | 36932 | 223 | -0.12% | +0.10% | 3.28 | -0.05% | 49% | -3.4% | -0.00% | +0.21% |
| `week52` sell1 | 493 | 94 | -1.06% | -0.01% | 1.24 | +0.04% | 51% | -3.9% | — | -0.01% |
| `week52` sell2 | 266 | 83 | -1.66% | -1.32% | -1.02 | -0.30% | 45% | -4.7% | — | -1.32% |
| `week52` sell3 | 21 | 18 | -3.72% | -3.34% | -2.15 | -2.12% | 33% | -8.3% | — | -3.34% |
| `liquidity` buy1 | 25690 | 223 | -0.26% | +0.06% | 0.80 | -0.05% | 49% | -3.6% | +0.04% | +0.09% |
| `liquidity` none | 12186 | 223 | -0.16% | +0.03% | -0.11 | -0.01% | 50% | -3.4% | +0.09% | -0.03% |
| `liquidity` sell1 | 12359 | 223 | -0.23% | -0.07% | -0.64 | -0.10% | 48% | -3.3% | -0.11% | -0.04% |
| `sector_rel` buy1 | 4940 | 222 | -0.81% | -0.62% | -3.25 | -0.79% | 38% | -4.6% | -0.55% | -0.69% |
| `sector_rel` none | 40921 | 223 | -0.12% | +0.12% | 5.73 | +0.04% | 51% | -3.3% | +0.12% | +0.13% |
| `sector_rel` sell1 | 4374 | 222 | -0.61% | -0.54% | -4.61 | -0.43% | 44% | -4.4% | -0.44% | -0.65% |

### Votes — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 46743 | 208 | -1.08% | -0.00% | — | -0.03% | 50% | -6.6% | -0.00% | -0.00% |
| `week52` buy1 | 5783 | 89 | -2.95% | -0.21% | -1.20 | +0.12% | 51% | -7.2% | — | -0.21% |
| `week52` buy2 | 4357 | 89 | -2.98% | -1.43% | -2.18 | -0.85% | 40% | -8.3% | — | -1.43% |
| `week52` none | 35921 | 208 | -0.49% | +0.36% | 2.20 | +0.05% | 51% | -6.2% | -0.00% | +0.80% |
| `week52` sell1 | 443 | 80 | -2.95% | +0.78% | -0.37 | +0.87% | 59% | -7.5% | — | +0.78% |
| `week52` sell2 | 221 | 68 | -4.84% | -1.37% | -3.57 | -0.44% | 45% | -9.1% | — | -1.37% |
| `week52` sell3 | 18 | 15 | -10.43% | -7.47% | — | -7.47% | 28% | -13.8% | — | -7.47% |
| `liquidity` buy1 | 24245 | 208 | -0.71% | +0.42% | 1.13 | +0.22% | 52% | -6.6% | +0.23% | +0.65% |
| `liquidity` none | 11236 | 208 | -1.11% | -0.20% | -0.74 | -0.13% | 48% | -6.5% | -0.06% | -0.36% |
| `liquidity` sell1 | 11262 | 208 | -1.83% | -0.66% | -1.37 | -0.40% | 45% | -6.5% | -0.81% | -0.48% |
| `sector_rel` buy1 | 4588 | 207 | -2.16% | -1.20% | -1.39 | -1.13% | 39% | -8.3% | -0.83% | -1.65% |
| `sector_rel` none | 38158 | 208 | -0.86% | +0.25% | 1.99 | +0.16% | 52% | -6.2% | +0.21% | +0.29% |
| `sector_rel` sell1 | 3997 | 207 | -1.97% | -1.33% | -3.74 | -0.93% | 41% | -8.1% | -0.91% | -1.84% |

## Replayed verdicts

### Verdicts — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 50235 | 223 | -0.23% | -0.00% | — | -0.05% | 49% | -3.5% | -0.00% | +0.00% |
| BUY | 77 | 47 | +0.13% | -0.26% | -2.32 | -0.57% | 45% | -4.2% | — | -0.26% |
| LEAN_BUY | 1691 | 104 | -0.20% | -0.27% | -0.71 | -0.17% | 46% | -4.0% | — | -0.27% |
| HOLD | 48426 | 223 | -0.23% | +0.01% | 1.11 | -0.04% | 49% | -3.5% | -0.00% | +0.02% |
| LEAN_SELL | 40 | 34 | -3.48% | -3.39% | -1.92 | -1.75% | 32% | -8.4% | — | -3.39% |
| SELL | 1 | 1 | -0.88% | +2.28% | — | +2.28% | 100% | -18.2% | — | +2.28% |

### Verdicts — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 46743 | 208 | -1.08% | -0.00% | — | -0.03% | 50% | -6.6% | -0.00% | -0.00% |
| BUY | 49 | 36 | -2.42% | -0.97% | — | -0.83% | 45% | -8.3% | — | -0.97% |
| LEAN_BUY | 1242 | 89 | -2.16% | -0.63% | -1.08 | -0.29% | 46% | -7.7% | — | -0.63% |
| HOLD | 45416 | 208 | -1.04% | +0.02% | 0.89 | -0.02% | 50% | -6.5% | -0.00% | +0.05% |
| LEAN_SELL | 35 | 29 | -9.79% | -7.01% | — | -5.19% | 23% | -14.2% | — | -7.01% |
| SELL | 1 | 1 | -17.19% | -17.71% | — | -17.71% | 0% | -19.7% | — | -17.71% |

Verdict mix over all replayed observations: HOLD 96%, LEAN_BUY 4%, BUY 0%, LEAN_SELL 0%, SELL 0%

## Feature information coefficients

Spearman rank correlation between each feature and the forward excess return, computed cross-sectionally per date and averaged. Positive IC = higher value, better future relative return. Q5−Q1 = top-quintile minus bottom-quintile excess return. Dates need ≥30 names.

| Feature | IC 5d | t 5d | Q5−Q1 5d | IC 20d | t 20d | Q5−Q1 20d | Dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| `drawdown_120` | 0.090 | 2.27 | +0.53% | 0.219 | 1.96 | +3.00% | 149 |
| `vol_20d` | -0.133 | -3.62 | -0.76% | -0.219 | -3.91 | -2.48% | 188 |
| `dist_sma120` | 0.100 | 1.63 | +0.60% | 0.216 | 1.18 | +2.81% | 89 |
| `atr14_pct` | -0.135 | -4.52 | -0.60% | -0.213 | -3.88 | -2.16% | 194 |
| `range_pos` | 0.052 | 1.04 | +0.35% | 0.149 | 1.78 | +1.75% | 89 |
| `dist_sma50` | 0.029 | 0.85 | +0.18% | 0.099 | 1.15 | +1.38% | 159 |
| `ret_60d` | 0.046 | 1.35 | +0.26% | 0.095 | 0.49 | +1.38% | 148 |
| `macd_hist_pct` | -0.048 | -1.66 | -0.33% | -0.036 | -0.44 | -0.25% | 175 |
| `ret_20d` | 0.009 | 0.36 | +0.09% | 0.035 | 0.24 | +0.57% | 188 |
| `sector_rel_20d` | 0.005 | 0.02 | +0.07% | 0.028 | -0.14 | +0.60% | 188 |
| `dist_sma20` | -0.026 | -0.99 | -0.16% | 0.026 | 0.37 | +0.40% | 189 |
| `rsi14` | -0.019 | -0.85 | -0.12% | 0.024 | 1.10 | +0.56% | 194 |
| `rturnover_20` | -0.057 | -3.63 | -0.50% | -0.023 | -0.80 | -0.30% | 188 |
| `ret_5d` | -0.056 | -1.60 | -0.35% | -0.016 | -1.13 | -0.11% | 203 |
| `rvol_20` | -0.048 | -3.02 | -0.44% | -0.009 | -0.35 | -0.18% | 188 |

## Quintiles — mean 20-day excess return (Q1 = lowest value)

| Feature | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| `range_pos` | -1.30% | -0.81% | +0.88% | +0.81% | +0.45% |
| `ret_20d` | -0.68% | +0.03% | +0.33% | +0.46% | -0.12% |
| `ret_60d` | -1.19% | -0.14% | +0.44% | +0.73% | +0.19% |
| `dist_sma50` | -1.10% | -0.18% | +0.23% | +0.79% | +0.28% |
| `rsi14` | -0.47% | -0.04% | +0.19% | +0.25% | +0.09% |
| `rvol_20` | -0.37% | +0.14% | +0.45% | +0.31% | -0.56% |
| `sector_rel_20d` | -0.80% | +0.12% | +0.41% | +0.49% | -0.20% |

## Scoring model (walk-forward)

Walk-forward weights start on **2026-01-07**, once 40 dates of fully realised 20-day outcomes exist. Everything in this section uses only those 168 out-of-sample dates (halves split at 2026-05-22).

- `score` = walk-forward weights: each day uses only ICs whose outcome had finished by then.
- `score_prior` = fixed prior weights chosen after looking at the full year in step 3 — **in-sample**, shown for reference only.
- Legacy = the replayed old vote engine on the same dates.

Weights (signed, sum of |w| = 1):

| Date | trend | momentum | low_risk | calm | liquidity |
|---|---:|---:|---:|---:|---:|
| prior | +0.30 | +0.10 | +0.30 | +0.20 | +0.10 |
| 2026-01-07 | +0.34 | +0.11 | +0.34 | +0.09 | +0.11 |
| 2026-03-23 | +0.47 | -0.14 | +0.16 | +0.13 | +0.10 |
| 2026-05-22 | +0.24 | -0.03 | +0.35 | +0.20 | +0.17 |
| 2026-07-23 | +0.24 | +0.03 | +0.39 | +0.20 | +0.14 |
| 2026-09-24 | +0.24 | +0.08 | +0.34 | +0.16 | +0.17 |

### Rank IC of each score vs forward excess return (out-of-sample dates)

| Score | IC 5d | t 5d | Q5−Q1 5d | IC 20d | t 20d | Q5−Q1 20d |
|---|---:|---:|---:|---:|---:|---:|
| Walk-forward score | 0.137 | 3.68 | +0.91% | 0.304 | 2.75 | +3.94% |
| Prior score (in-sample) | 0.147 | 3.81 | +0.99% | 0.311 | 2.85 | +3.99% |
| Risk score (higher = riskier) | -0.153 | -3.20 | -0.77% | -0.280 | -3.36 | -2.91% |
| Legacy buy − sell | -0.021 | -0.22 | -0.05% | -0.006 | 0.72 | -0.14% |

### Classifications — 5-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 36693 | 162 | -0.25% | +0.00% | — | -0.04% | 49% | -3.6% | +0.00% | +0.00% |
| STRONG_SETUP | 3720 | 162 | +0.29% | +0.54% | 2.84 | +0.46% | 61% | -2.4% | +0.43% | +0.65% |
| SETUP | 5260 | 162 | +0.07% | +0.31% | 1.90 | +0.22% | 56% | -2.8% | +0.20% | +0.43% |
| WATCH | 5172 | 162 | -0.07% | +0.18% | 2.05 | +0.07% | 52% | -3.1% | +0.13% | +0.23% |
| NEUTRAL | 13256 | 162 | -0.28% | -0.04% | -0.34 | -0.15% | 47% | -3.6% | -0.06% | -0.02% |
| AVOID | 6321 | 162 | -0.62% | -0.32% | -2.30 | -0.36% | 44% | -4.2% | -0.19% | -0.45% |
| HIGH_RISK | 2964 | 162 | -0.89% | -0.66% | -1.88 | -0.83% | 39% | -5.5% | -0.48% | -0.85% |
| STRONG_SETUP (prior weights, in-sample) | 3676 | 162 | +0.30% | +0.56% | 2.73 | +0.50% | 62% | -2.4% | +0.47% | +0.65% |
| SETUP (prior weights, in-sample) | 5243 | 162 | +0.08% | +0.32% | 2.57 | +0.22% | 55% | -2.8% | +0.22% | +0.43% |
| Legacy BUY | 77 | 47 | +0.13% | -0.26% | -2.32 | -0.57% | 45% | -4.2% | +0.59% | -0.49% |
| Legacy LEAN_BUY | 1691 | 104 | -0.20% | -0.27% | -0.71 | -0.17% | 46% | -4.0% | -0.99% | -0.03% |

### Classifications — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All observations (baseline) | 33201 | 147 | -1.88% | -0.00% | — | +0.10% | 51% | -7.2% | -0.00% | -0.00% |
| STRONG_SETUP | 3364 | 147 | +0.40% | +2.30% | 2.48 | +2.13% | 75% | -4.6% | +1.85% | +2.91% |
| SETUP | 4768 | 147 | -0.36% | +1.55% | 2.30 | +1.29% | 66% | -5.5% | +1.24% | +1.96% |
| WATCH | 4692 | 147 | -1.28% | +0.63% | 1.60 | +0.57% | 58% | -6.3% | +0.34% | +1.03% |
| NEUTRAL | 12045 | 147 | -2.20% | -0.35% | -1.45 | -0.28% | 47% | -7.3% | -0.23% | -0.50% |
| AVOID | 5650 | 147 | -3.36% | -1.40% | -1.59 | -1.28% | 37% | -8.7% | -1.22% | -1.64% |
| HIGH_RISK | 2682 | 147 | -3.98% | -2.24% | -2.67 | -2.35% | 34% | -10.8% | -1.57% | -3.13% |
| STRONG_SETUP (prior weights, in-sample) | 3332 | 147 | +0.46% | +2.35% | 1.59 | +2.22% | 76% | -4.5% | +1.92% | +2.93% |
| SETUP (prior weights, in-sample) | 4756 | 147 | -0.47% | +1.41% | 3.64 | +1.15% | 65% | -5.3% | +1.11% | +1.81% |
| Legacy BUY | 49 | 36 | -2.42% | -0.97% | — | -0.83% | 45% | -8.3% | +0.31% | -1.46% |
| Legacy LEAN_BUY | 1242 | 89 | -2.16% | -0.63% | -1.08 | -0.29% | 46% | -7.7% | -0.58% | -0.64% |

### Market regime

Regime days in the out-of-sample window: BEARISH 75, NEUTRAL 60, BULLISH 33. Would skipping STRONG_SETUP names in BEARISH regimes have avoided losses? Judge on **raw** returns (excess is market-neutral). The live pipeline shows the regime as context but does not gate on it: the evidence is inconclusive — few BULLISH days, and they were not better than BEARISH ones.

### Regime gate — 20-day horizon

| Group | Obs | Dates | Raw ret | Excess | t (non-overlap) | Median excess | Hit rate | Avg max DD | Excess 1st half | Excess 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STRONG_SETUP in BULLISH | 736 | 33 | -0.65% | +2.25% | — | +1.89% | 71% | -7.4% | +2.25% | — |
| STRONG_SETUP in NEUTRAL | 1278 | 56 | +1.98% | +2.36% | 1.35 | +2.30% | 73% | -3.6% | +1.93% | +3.52% |
| STRONG_SETUP in BEARISH | 1350 | 58 | -0.53% | +2.27% | 2.91 | +2.10% | 80% | -4.1% | +0.14% | +2.72% |
| STRONG_SETUP, all regimes | 3364 | 147 | +0.40% | +2.30% | 2.48 | +2.13% | 75% | -4.6% | +1.85% | +2.91% |
| STRONG_SETUP excluding BEARISH (gate) | 2014 | 89 | +1.02% | +2.32% | 2.40 | +2.14% | 72% | -5.0% | +2.08% | +3.52% |

