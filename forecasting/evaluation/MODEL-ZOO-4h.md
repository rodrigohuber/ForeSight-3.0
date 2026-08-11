# Model zoo — 4h realized volatility

Walk-forward, 15 folds, **31,382 scored hours**, 46 features. Every model refit on each expanding window and scored on the block that follows.

| model | RMSE | MAE | QLIKE | R² vs naive | R² vs HAR | fit time |
|---|---|---|---|---|---|---|
| **LightGBM** | 0.3087 | 0.2305 | 0.2847 | +0.8773 | +0.3011 | 21s |
| XGBoost | 0.3170 | 0.2368 | 0.2821 | +0.8706 | +0.2631 | 25s |
| MLP (64-32) | 0.3275 | 0.2466 | 0.3275 | +0.8619 | +0.2134 | 195s |
| Random Forest | 0.3285 | 0.2500 | 0.3233 | +0.8610 | +0.2086 | 191s |
| Extra Trees | 0.3371 | 0.2596 | 0.3376 | +0.8536 | +0.1665 | 38s |
| MLP (256-128-64) | 0.3404 | 0.2592 | 0.3365 | +0.8508 | +0.1502 | 704s |
| Ridge | 0.3503 | 0.2733 | 0.3686 | +0.8419 | +0.0999 | 6s |
| HAR | 0.3692 | 0.2910 | 0.3996 | +0.8244 | +0.0000 | — |
| naive | 0.8811 | 0.7746 | 4.7794 | +0.0000 | -4.6944 | — |

## Is **LightGBM** significantly better than the rest?

| challenger | DM stat | p-value | verdict |
|---|---|---|---|
| vs XGBoost | -6.26 | 3.8e-10 | LightGBM better |
| vs MLP (64-32) | -11.94 | 0.0e+00 | LightGBM better |
| vs Random Forest | -17.52 | 0.0e+00 | LightGBM better |
| vs Extra Trees | -22.21 | 0.0e+00 | LightGBM better |
| vs MLP (256-128-64) | -16.47 | 0.0e+00 | LightGBM better |
| vs Ridge | -21.50 | 0.0e+00 | LightGBM better |
| vs HAR | -26.29 | 0.0e+00 | LightGBM better |
| vs naive | -139.90 | 0.0e+00 | LightGBM better |

