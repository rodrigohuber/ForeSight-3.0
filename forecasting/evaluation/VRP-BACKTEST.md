# Variance risk premium — is it profitable after costs?

Monthly non-overlapping entries: **43 trades**, 2023-01 .. 2026-07.
Round-trip cost assumed: **1.5 vol points**.

P&L is quoted in *vol points* per trade: short variance earns
(implied − realized) − cost. Positive means the premium was collected.

| strategy | trades | total | mean/trade | win rate | worst | Sharpe | max DD |
|---|---|---|---|---|---|---|---|
| A · always short | 43 | +1.1 | +0.02 | 58% | -28.2 | 0.01 | 89.9 |
| B · short only if edge ≥ 2.0 | 16 | +29.3 | +1.83 | 69% | -12.7 | 0.75 | 12.7 |

Gross premium before costs: +1.52 vol points per trade (60% of trades positive).

## How much cost kills it?

| round-trip cost | mean net | Sharpe | win rate |
|---|---|---|---|
| 0.0 vol pts | +1.52 | 0.46 | 60% |
| 1.0 vol pts | +0.52 | 0.16 | 58% |
| 1.5 vol pts | +0.02 | 0.01 | 58% |
| 2.0 vol pts | -0.48 | -0.14 | 51% |
| 3.0 vol pts | -1.48 | -0.45 | 49% |
| 4.0 vol pts | -2.48 | -0.75 | 47% |
| 5.0 vol pts | -3.48 | -1.06 | 47% |

## The tail, which is the whole risk

Five worst trades, gross: -26.7, -23.4, -20.8, -16.7, -12.3 vol points.
Mean of the worst 10%: -21.9.

Short volatility loses most when volatility explodes, which is also
when a delta-hedged implementation bleeds hedging costs and when margin
is called. The proxy used here charges neither.

