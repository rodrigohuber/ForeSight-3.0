# Positioning as a directional signal — 4h bars

**7,101 bars**, 2023-01-30 .. 2026-07-31. Binance USD-M futures metrics, free, published every five minutes.

## Does any of it predict the next bar?

Spearman against the next bar's return, Benjamini-Hochberg corrected across all 16 features.

| feature | ρ | raw p | BH p | survives? |
|---|---|---|---|---|
| oi | -0.0279 | 0.019 | 0.214 | no |
| oi_z | -0.0263 | 0.027 | 0.214 | no |
| oi_chg | -0.0147 | 0.217 | 0.927 | no |
| smart_minus_crowd | +0.0139 | 0.242 | 0.927 | no |
| toptrader_pos | -0.0118 | 0.318 | 0.927 | no |
| all_accounts_chg | -0.0107 | 0.368 | 0.927 | no |
| toptrader_pos_chg | +0.0099 | 0.406 | 0.927 | no |
| toptrader_acct_chg | +0.0076 | 0.523 | 0.940 | no |
| toptrader_pos_z | +0.0051 | 0.668 | 0.940 | no |
| toptrader_acct_z | +0.0050 | 0.675 | 0.940 | no |

**0 of 16 features survive correction.**

## Quintile detail on the strongest candidate (`oi`)

| bucket | mean next-bar return | n |
|---|---|---|
| lowest | +0.0343% | 1,421 |
| 2 | +0.0098% | 1,420 |
| 3 | +0.0650% | 1,420 |
| 4 | -0.0508% | 1,420 |
| highest | -0.0140% | 1,420 |

Top-minus-bottom spread: **-0.0483%** per bar, against a round-trip cost of 0.20%.
→ **does NOT clear its own transaction cost.**

## As a strategy, after costs

| rule | total return | Sharpe | max DD |
|---|---|---|---|
| buy & hold | +87.3% | 0.42 | 0.79 |
| long when oi high | -36.9% | -0.47 | 1.03 |
| long when oi low | +82.2% | 0.54 | 0.35 |

⚠ Sharpe standard error here is about 0.56, and this sample is only 2023 onward because Binance retains metrics files for a limited window.

