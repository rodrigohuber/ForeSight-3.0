# Phase 2 — jump-robust features, multi-horizon, quantiles

Walk-forward evaluation: expanding window, refit every 90 days, each
block scored before it is ever trained on. Every number below is
out-of-sample by construction.

## 1. Does feature set v2 beat v1?

v1: 27 features, 31,388 scored hours
v2: 46 features, 31,388 scored hours

| feature set | RMSE | QLIKE | R2 vs HAR |
|---|---|---|---|
| v1 (phase 1) | 0.3260 | 0.3490 | +0.1721 |
| v2 (jump + HARQ) | 0.3232 | 0.3532 | +0.1865 |

## 2. Does the edge hold at longer horizons?

| horizon | hours scored | HAR RMSE | LightGBM RMSE | R2 vs HAR | DM p |
|---|---|---|---|---|---|
| 1h | 31,388 | 0.3583 | 0.3232 | +0.1865 | 0.0e+00 |
| 4h | 31,382 | 0.3692 | 0.3087 | +0.3011 | 0.0e+00 |
| 24h | 31,342 | 0.3376 | 0.2866 | +0.2795 | 0.0e+00 |

## 3. The tail — quantile forecasts

Phase 1's point forecast missed the largest spike by 5x. A point
forecast cannot express 'this hour could be violent'; an upper
quantile can. Scored with pinball loss and empirical coverage.

| quantile | pinball loss | empirical coverage | target |
|---|---|---|---|
| 0.05 | 0.02782 | 0.067 | 0.05 |
| 0.50 | 0.11891 | 0.489 | 0.50 |
| 0.95 | 0.04360 | 0.931 | 0.95 |

**90% interval covers 86.5% of realized hours** (target 90%).

Of the worst 1% of hours, the 0.95-quantile model's own top-1% forecasts flag **32.5%**.

