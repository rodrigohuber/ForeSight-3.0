# Technical analysis on BTC — D bars

**3,020 bars**, 38 indicators, 2018-04-24 .. 2026-07-30. Round-trip cost 5 bps.

## 1. Do individual indicators predict the next bar?

Spearman correlation between each indicator and the NEXT bar's return,
with a Benjamini-Hochberg correction across all indicators tested.

| indicator | Spearman ρ | raw p | BH-adjusted p | survives? |
|---|---|---|---|---|
| roc1 | -0.0609 | 0.001 | 0.031 | **yes** |
| stoch_kd | -0.0450 | 0.013 | 0.184 | no |
| body_frac | -0.0445 | 0.015 | 0.184 | no |
| roc60 | +0.0375 | 0.039 | 0.374 | no |
| px_vs_sma100 | +0.0347 | 0.057 | 0.430 | no |
| di_spread | +0.0320 | 0.079 | 0.432 | no |
| sma20_vs_50 | +0.0319 | 0.080 | 0.432 | no |
| roc120 | +0.0293 | 0.108 | 0.481 | no |

**1 of 38 indicators survive multiple-testing correction.**

## 2. A model given every indicator at once

Walk-forward, 2,037 bars scored (2021-01-01 .. 2026-07-30).

| metric | value |
|---|---|
| R² vs predicting zero | -0.0640 |
| directional hit rate | 0.4953 |
| correlation forecast vs realized | +0.0051 |

## 3. Strategies, after costs

| strategy | total return | Sharpe | hit | max drawdown |
|---|---|---|---|---|
| buy & hold | +114.4% | 0.24 | 50% | 1.45 |
| golden cross (50/200) | +101.9% | 0.29 | 27% | 0.84 |
| TS momentum, vol-scaled | +290.2% | 0.89 | 52% | 0.46 |
| ML on all indicators | -25.4% | -0.13 | 25% | 1.21 |

⚠ Two things to read carefully. Sharpe on this few bars carries a standard error of roughly 0.42, so differences smaller than that are noise. And the *hit* column is deflated for long/flat strategies: a flat day returns exactly zero, which does not count as a win, so a strategy that stands aside half the time cannot score above ~50% however good it is.

