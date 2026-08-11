# GARCH family vs HAR vs LightGBM

Walk-forward, 15 folds, **31,388 scored hours**, next-hour log realized volatility.

GARCH parameters are estimated once per fold on the preceding 8,000
hours, then held fixed while the conditional variance is filtered
forward. That is standard practice and avoids refitting hourly.

| model | RMSE | MAE | QLIKE | R² vs naive | R² vs HAR |
|---|---|---|---|---|---|
| **LightGBM** | 0.3232 | 0.2375 | 0.3532 | +0.2965 | +0.1865 |
| HAR | 0.3583 | 0.2706 | 0.4315 | +0.1352 | +0.0000 |
| naive | 0.3853 | 0.2897 | 0.4905 | +0.0000 | -0.1563 |
| GARCH(1,1) | 0.5607 | 0.4541 | 0.5941 | -1.1176 | -1.4487 |
| GJR-GARCH | 0.5609 | 0.4544 | 0.5935 | -1.1193 | -1.4507 |
| EGARCH(1,1) | 0.5864 | 0.4817 | 0.5683 | -1.3162 | -1.6783 |

## Diebold-Mariano, LightGBM against each

| challenger | DM | p | verdict |
|---|---|---|---|
| vs HAR | -29.30 | 0.0e+00 | LightGBM better |
| vs naive | -42.88 | 0.0e+00 | LightGBM better |
| vs GARCH(1,1) | -41.38 | 0.0e+00 | LightGBM better |
| vs GJR-GARCH | -41.24 | 0.0e+00 | LightGBM better |
| vs EGARCH(1,1) | -45.05 | 0.0e+00 | LightGBM better |

