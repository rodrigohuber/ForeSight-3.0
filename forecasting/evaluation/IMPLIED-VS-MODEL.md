# Implied vs realized — does the model beat the market?

Aligned hours: **32,144**  (2021-04-11 .. 2026-07-02)

DVOL is Deribit's 30-day BTC implied volatility, annualized, in percent.
Realized is annualized volatility over the following 30 days, computed
from 1-minute bars, coverage-checked at 95%.

## 1. Variance risk premium

| quantity | value |
|---|---|
| mean implied (DVOL) | 59.99% |
| mean subsequent realized | 55.78% |
| **mean premium (implied − realized)** | **+4.20 vol points** |
| share of windows where implied exceeded realized | 67.5% |
| Newey-West t-stat (720h lag) | 2.24 |

**Implied volatility sits systematically above what follows.** That is
the variance risk premium, and harvesting it requires no forecasting
skill — only the willingness to be short volatility and survive the
windows where it inverts.

## 2. Who forecasts 30-day realized volatility better?

Walk-forward, 21,433 scored hours (2023-01-01 .. 2026-07-02). All in log space.

| forecaster | RMSE | bias |
|---|---|---|
| DVOL (the market) | 0.2265 | +0.0658 |
| our model | 0.3107 | +0.0795 |

## 3. Does our forecast add anything the market has not priced?

Mincer-Zarnowitz encompassing regression of realized on the forecasts:

| specification | R² | coefficient on our forecast |
|---|---|---|
| realized ~ DVOL | 0.2327 | — |
| realized ~ DVOL + ours | 0.2525 | -0.1770 |

Incremental R² from our forecast: **+0.0197**

⚠ With 30-day windows sampled hourly, consecutive rows share 719 of 720
hours. The effective sample is nearer the ~64 non-overlapping months than
the row count, so treat any t-statistic here as indicative, not decisive.

