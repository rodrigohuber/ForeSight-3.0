# Phase 1 — next-hour realized volatility

Features: 27 | rows: inner-train 46,753, inner-val 8,757, validation 13,128, test 9,503

## Validation (model selection)

`2024-01-01` .. `2025-06-30`, 13,128 hours

| model | RMSE | MAE | QLIKE | R2 vs naive | R2 vs HAR |
|---|---|---|---|---|---|
| naive | 0.3552 | 0.2685 | 0.3781 | +0.0000 | -0.0987 |
| HAR | 0.3388 | 0.2603 | 0.3405 | +0.0898 | +0.0000 |
| Ridge | 0.3247 | 0.2471 | 0.3156 | +0.1643 | +0.0819 |
| MLP | 0.3273 | 0.2453 | 0.3408 | +0.1509 | +0.0671 |
| LightGBM | 0.3126 | 0.2368 | 0.2838 | +0.2255 | +0.1490 |

### Diebold-Mariano (squared-error loss, Newey-West HAC)

A lower RMSE is not automatically better; the gap must exceed the noise
in the loss differential.

| comparison | DM stat | p-value | verdict |
|---|---|---|---|
| LightGBM vs HAR | -18.53 | 0.00e+00 | **LightGBM** better |
| LightGBM vs MLP | -10.42 | 0.00e+00 | **LightGBM** better |
| MLP vs HAR | -6.84 | 8.06e-12 | **MLP** better |
| Ridge vs HAR | -16.09 | 0.00e+00 | **Ridge** better |
| HAR vs naive | -8.85 | 0.00e+00 | **HAR** better |

### Robustness: per-quarter R2 vs HAR

One validation window can flatter a model. If the edge is real it
should survive being cut into pieces.

| quarter | hours | LightGBM | MLP | Ridge |
|---|---|---|---|---|
| 2024Q1 | 2,184 | +0.1630 | +0.1282 | +0.0878 |
| 2024Q2 | 2,184 | +0.1091 | +0.0481 | +0.0398 |
| 2024Q3 | 2,208 | +0.1665 | +0.0723 | +0.0846 |
| 2024Q4 | 2,208 | +0.1846 | +0.0603 | +0.1051 |
| 2025Q1 | 2,160 | +0.1699 | +0.0616 | +0.1080 |
| 2025Q2 | 2,184 | +0.1051 | +0.0281 | +0.0725 |

**LightGBM beats HAR in 6 of 6 quarters.**

---

## Test set — read after the model was locked

LightGBM is selected on the validation evidence above.

> ⚠ **Disclosure: the test window has been read twice.** The first pass
> exposed a *feature* defect, not a model result: `semivar_ratio` was an
> unbounded rv_neg/rv_pos, reaching 9.3 in train but 1,929 in test, and
> the scaled Ridge and MLP extrapolated into RMSE 1.8 and 3.4. It was
> replaced with a bounded downside-variance share. The fix was made on
> grounds independent of test performance — an unbounded ratio is a
> defect whatever it scores — and no hyperparameter was changed. The
> numbers below are the second pass. Discount them accordingly: they
> are no longer a virgin test set, and the honest place to re-earn one
> is new out-of-sample data as it arrives.

`2025-07-01` .. `2026-07-31`, 9,503 hours

| model | RMSE | MAE | QLIKE | R2 vs naive | R2 vs HAR |
|---|---|---|---|---|---|
| naive | 0.3666 | 0.2763 | 0.4151 | +0.0000 | -0.1328 |
| HAR | 0.3444 | 0.2619 | 0.3579 | +0.1172 | +0.0000 |
| Ridge | 0.3340 | 0.2526 | 0.3353 | +0.1698 | +0.0596 |
| MLP | 0.3516 | 0.2607 | 0.4283 | +0.0802 | -0.0419 |
| LightGBM | 0.3249 | 0.2453 | 0.3098 | +0.2145 | +0.1102 |

Diebold-Mariano, LightGBM vs HAR on test: DM=-13.33, p=0.00e+00

