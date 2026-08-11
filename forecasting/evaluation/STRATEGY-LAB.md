# Strategy lab — momentum and volatility targeting

**2,905 daily bars**, 2018-08-17 .. 2026-07-30. Signals from closes only; the earlier lookahead bug is gone.

Vol targeting scales position to a 40% annualised target, capped at 3x leverage.

## 1. Signal or sizing — which is doing the work?

Costs at 10 bps round trip. `always long` isolates the sizing effect;
`constant size` isolates the momentum effect. The `model` sizing rows
cover 2021 onward only (2,037 bars), since the forecast needs a walk-forward history first; the others use all 2,905 bars.

| position rule | sizing | bars | total | Sharpe | max DD | turnover/yr |
|---|---|---|---|---|---|---|
| always long | constant | 2,905 | +854.1% | 0.45 | 1.45 | 0.1x |
| always long | ATR | 2,905 | +449.2% | 0.45 | 1.15 | 7.2x |
| always long | model | 2,037 | +17.8% | 0.07 | 1.06 | 71.8x |
| momentum 365d | constant | 2,905 | +1324.0% | 0.63 | 1.00 | 3.3x |
| momentum 365d | ATR | 2,905 | +510.2% | 0.61 | 0.66 | 7.2x |
| momentum 365d | model | 2,037 | +156.6% | 0.50 | 0.49 | 47.8x |

Buy-and-hold reference: +854.1%, Sharpe 0.45, max drawdown 1.45.

⚠ Compare `model` rows only against each other -- they run on a different, shorter window than the rest of the table.

## 2. Every lookback, every cost — the whole grid

Reported in full so nothing is cherry-picked. Sizing = ATR vol target.

| lookback | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| 30d | 1.09 | 1.04 | 0.99 | 0.90 |
| 60d | 0.98 | 0.95 | 0.92 | 0.86 |
| 90d | 0.65 | 0.63 | 0.61 | 0.56 |
| 180d | 0.61 | 0.60 | 0.59 | 0.56 |
| 365d | 0.63 | 0.62 | 0.61 | 0.59 |

Cells are Sharpe ratios. Buy-and-hold with the same vol targeting scores 0.45 at 10 bps.

## 3. Does it survive being cut into regimes?

Every lookback, ATR vol targeting, 10 bps, against buy-and-hold sized
the same way. Sharpe ratios.

| lookback | 2018-2020 | 2021-2022 | 2023-2026 | full |
|---|---|---|---|---|
| 30d | 1.67 | -0.38 | 1.08 | **0.99** |
| 60d | 1.71 | -0.38 | 0.97 | **0.92** |
| 90d | 1.04 | -0.55 | 0.75 | **0.61** |
| 180d | 1.16 | -0.33 | 0.61 | **0.59** |
| 365d | 0.66 | 0.03 | 0.85 | **0.61** |
| *buy & hold* | 0.90 | -0.85 | 0.80 | **0.45** |

Drawdowns, same configurations:

| lookback | max drawdown |
|---|---|
| 30d | 0.54 |
| 60d | 0.72 |
| 90d | 0.71 |
| 180d | 0.73 |
| 365d | 0.66 |
| *buy & hold* | 1.15 |

⚠ Sharpe standard error on the full sample is about 0.35. Sub-period figures are far noisier still. Treat any single cell as indicative and the pattern across cells as the evidence.

