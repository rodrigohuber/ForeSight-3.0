# Session log — 2026-08-06

Objective as given: *"make a forecast model that works and takes profit."* Then,
after the first results: *"do them all in sequence"* — validate, extend, write up.

**Outcome: seven things tested, six ended, one survives.** The survivor is a
well-known risk premium, not a prediction. That distinction is the session's
main finding.

---

## The scoreboard

| tested | verdict | evidence |
|---|---|---|
| Minute-scale direction from price | **DEAD** | R² −1.02 vs predicting zero, 49.4% hit rate |
| Classic chart indicators (38 of them) | **DEAD** | 1 survives BH correction, and points the wrong way |
| GARCH / GJR-GARCH / EGARCH | **BEATEN** | All three fall *below* the naive carry-forward |
| Our vol model vs the options market | **LOST** | DVOL RMSE 0.227 vs our 0.311 |
| Variance risk premium after costs | **BREAKEVEN** | Breakeven at ~1.5 vol points round-trip |
| Funding carry (long spot / short perp) | **ARBITRAGED** | 30.6% in 2021 → 3.4% trailing twelve months |
| Positioning data (OI, top-trader ratios) | **DEAD** | 0 of 16 features survive correction |
| **Trend following + vol-scaled sizing** | **SURVIVES** | Beats buy-and-hold on BTC *and* ETH |

---

## The one that survived

Long when the trailing return is positive, sized inversely to recent
volatility, otherwise flat. Daily bars, 10 bps costs.

| lookback | BTC | ETH | BTC max DD |
|---|---|---|---|
| 30d | **0.99** | **0.81** | 0.54 |
| 60d | 0.92 | 0.73 | 0.72 |
| 90d | 0.61 | 0.49 | 0.71 |
| 180d | 0.59 | 0.31 | 0.73 |
| 365d | 0.61 | 0.40 | 0.66 |
| *buy & hold* | *0.45* | *0.25* | *1.15* |

Every lookback beats buy-and-hold **on both assets**, at every cost level to
20 bps, with drawdowns roughly halved. The value is concentrated in downside
protection: through 2021–22 every variant lost less than holding.

**The ETH replication is what made this credible.** The rank ordering across
lookbacks is nearly identical on two independent assets. A pattern that exists
only on Bitcoin is a Bitcoin fact.

---

## Three findings that changed what I believed

### 1. The first momentum result was contaminated
The signal was built from `cumsum` of **forward** returns, so the running total
at *t* already contained the return being predicted — correlation exactly
**1.000**. The 12-month sign only flipped on 0.8% of days so the conclusion
survived rebuilding, but it had to be measured, not assumed.

### 2. Volatility targeting does not raise Sharpe — the signal does
0.45 either way for buy-and-hold. It buys **drawdown reduction** (1.45 → 1.15),
not return. Credit for trend-following performance usually goes to the sizing;
here it belongs to the signal.

### 3. ⚠ A better forecast is a worse position sizer
Our LightGBM vol model beats HAR by 30% on accuracy. As a sizer:

| sizing | momentum Sharpe | turnover/yr |
|---|---|---|
| simple 20-day realized vol | **0.61** | 7× |
| our LightGBM forecast | 0.50 | 48× |

For always-long it is worse still: 0.07 vs 0.45. **Forecasting rewards
responsiveness; sizing rewards stability.** Days were spent optimising the
wrong objective, and no additional accuracy would have fixed it.

---

## What the funding data showed

Free, published every eight hours, and the clearest picture of market
efficiency in the whole project:

| year | annualised funding |
|---|---|
| 2020 | +17.2% |
| **2021** | **+30.6%** |
| 2022 | +4.2% |
| 2023 | +7.9% |
| 2024 | +11.9% *(spot-ETF launch)* |
| 2025 | +5.1% |
| 2026* | +1.9% |

Trailing twelve months: **+3.4%**. A government bond pays comparably without
exchange counterparty risk, margin capital, or liquidation risk.

The naive carry Sharpe computes to **17**, which should be disbelieved on
sight — it assumes a perfect free delta hedge. At 5% hedge tracking error it is
2.35; at 10%, 1.18.

---

## Honest limits on everything above

- **Sharpe standard error is 0.35.** The trend margin over buy-and-hold is
  roughly 0.5–1.5 SE. Suggestive, not conclusive.
- Both assets rose enormously across the window, flattering a long-biased rule.
- Trend following is a **crowded, documented premium**, not a discovery.
- A backtest without slippage, funding or execution risk is not a live result.
- The positioning sample starts 2023 (Binance retains metrics files for a
  limited window only).

---

## Method discipline applied throughout

Chronological splits only; walk-forward with every block scored before it is
trained on; costs charged everywhere; Benjamini-Hochberg correction when
testing 38 indicators at once; the **full parameter grid reported rather than
its best cell**; every source archive SHA-256 verified against the checksum
published beside it.

Two bugs produced believable wrong answers and were caught by results being
*implausible*, not by tests failing: the December-2017 "14-day outage" (20,400
real bars with sub-minute timestamps) and the forward-return leak above.

---

## Deliverables

| artefact | what |
|---|---|
| `PHASE1-REPORT.md` | Volatility forecasting, phase 1 |
| `forecasting/evaluation/*.md` | 9 result files, all script-generated |
| `analysis/reports/final-report-template.html` | The shareable write-up source |
| `scripts/forecast_now.py` | Live forecast + logging + `--score` |
| `scripts/train_production.py` | Saves models with a provenance manifest |

**27 Python modules · 24 commits · 9.4M minute bars across BTC and ETH.**

---

## Commits

`8998018a` strategy lab (leak fixed) · `42fc282c` first session log ·
`5630374e` futures ingester · `3df2e206` funding analysis ·
`f9e9a7ea` ETH replication · `3084c28f` final report ·
`d854138b` positioning test

Earlier the same day: `790bf41b` model zoo · `524cee38` technical analysis ·
`b03f5667` GARCH + VRP · `9cb96cf6` implied vs model.

---

## If this is picked up again

1. **Do not** keep optimising volatility forecast accuracy — finding 3 shows it
   does not transfer to strategy performance.
2. The trend result needs **live paper-trading** before it needs more backtests.
   Another parameter sweep adds nothing; contact with real slippage does.
3. Everything untested that remains is expensive: L2 order book (needs
   latency), on-chain (needs a paid feed), options surface beyond ATM.
4. **The opportunity cost is real.** This project has produced its insights;
   the next marginal hour is worth more elsewhere until paper trading runs.
