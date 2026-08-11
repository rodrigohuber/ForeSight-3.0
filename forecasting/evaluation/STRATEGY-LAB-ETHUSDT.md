# Strategy lab — ETHUSDT

**2,905 daily bars**, 2018-08-17 .. 2026-07-30. Signals from closes only; the earlier lookahead bug is gone.

Vol targeting scales position to a 40% annualised target, capped at 3x leverage.

## 1. Signal or sizing — which is doing the work?

Costs at 10 bps round trip. `always long` isolates the sizing effect;
`constant size` isolates the momentum effect. The `model` sizing rows
cover 2021 onward only (0 bars), since the forecast needs a walk-forward history first; the others use all 2,905 bars.

| position rule | sizing | bars | total | Sharpe | max DD | turnover/yr |
|---|---|---|---|---|---|---|
| always long | constant | 2,905 | +486.0% | 0.26 | 1.58 | 0.1x |
| always long | ATR | 2,905 | +157.9% | 0.25 | 0.93 | 5.3x |
| always long | model | 0 | +nan% | nan | nan | nanx |
| momentum 365d | constant | 2,905 | +448.0% | 0.34 | 1.20 | 6.8x |
| momentum 365d | ATR | 2,905 | +197.1% | 0.40 | 0.56 | 7.7x |
| momentum 365d | model | 0 | +nan% | nan | nan | nanx |

Buy-and-hold reference: +486.0%, Sharpe 0.26, max drawdown 1.58.

⚠ Compare `model` rows only against each other -- they run on a different, shorter window than the rest of the table.

## 2. Every lookback, every cost — the whole grid

Reported in full so nothing is cherry-picked. Sizing = ATR vol target.

| lookback | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| 30d | 0.87 | 0.84 | 0.81 | 0.74 |
| 60d | 0.77 | 0.75 | 0.73 | 0.69 |
| 90d | 0.52 | 0.51 | 0.49 | 0.45 |
| 180d | 0.34 | 0.33 | 0.31 | 0.28 |
| 365d | 0.43 | 0.41 | 0.40 | 0.38 |

Cells are Sharpe ratios. Buy-and-hold with the same vol targeting scores 0.25 at 10 bps.

## 3. Does it survive being cut into regimes?

Every lookback, ATR vol targeting, 10 bps, against buy-and-hold sized
the same way. Sharpe ratios.

| lookback | 2018-2020 | 2021-2022 | 2023-2026 | full |
|---|---|---|---|---|
| 30d | 1.07 | 0.70 | 0.67 | **0.81** |
| 60d | 1.13 | 0.38 | 0.62 | **0.73** |
| 90d | 0.90 | 0.16 | 0.36 | **0.49** |
| 180d | 0.73 | 0.66 | -0.16 | **0.31** |
| 365d | 0.27 | 0.60 | 0.38 | **0.40** |
| *buy & hold* | 0.55 | -0.09 | 0.23 | **0.25** |

Drawdowns, same configurations:

| lookback | max drawdown |
|---|---|
| 30d | 0.44 |
| 60d | 0.47 |
| 90d | 0.64 |
| 180d | 0.93 |
| 365d | 0.56 |
| *buy & hold* | 0.93 |

⚠ Sharpe standard error on the full sample is about 0.35. Sub-period figures are far noisier still. Treat any single cell as indicative and the pattern across cells as the evidence.

