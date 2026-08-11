# Funding rate — signal, and income

7,212 funding settlements, 2020-01-01 .. 2026-07-31, every 8 hours.

| statistic | value |
|---|---|
| mean funding per 8h | +0.0108% |
| annualised | **+11.9%** |
| intervals positive | 85.7% |
| median | +0.0094% |
| worst single settlement | -0.300% |

Positive funding means longs pay shorts. It has been positive in the
large majority of intervals, which is a structural cost of holding
leveraged long exposure and a structural income for the other side.

## 1. Does funding predict the next move?

| predictor | Spearman ρ | p-value | reading |
|---|---|---|---|
| funding level | -0.0118 | 0.460 | no reliable relationship |
| funding z-score (90p) | -0.0408 | 0.011 | contrarian, as theory predicts |

Next-8h return by funding z-score bucket:

| funding bucket | mean next 8h return | n |
|---|---|---|
| most negative | +0.1550% | 783 |
| 2 | +0.1436% | 782 |
| 3 | +0.0481% | 782 |
| 4 | +0.0745% | 782 |
| most positive | +0.0179% | 783 |

Top-minus-bottom bucket spread: **-0.1371%** per 8 hours. A round trip costs 0.20%, so the spread would need to exceed that to be tradeable as a signal.

## 2. The cash-and-carry trade — income without a forecast

Long spot, short the perpetual, net delta about zero. Collect funding
when positive, pay it when negative. No view on direction required.

| entry+exit cost (both legs) | gross annualised | net annualised |
|---|---|---|
| 0 bps | +11.9% | +11.9% |
| 10 bps | +11.9% | +11.8% |
| 20 bps | +11.9% | +11.8% |
| 50 bps | +11.9% | +11.7% |

Costs are one-off on a continuously held position, so they barely dent
a multi-year hold. The risk is not cost, it is that funding inverts.

| carry-stream risk | value |
|---|---|
| worst drawdown in cumulative funding | 1.52% |
| worst calendar year | +1.1% |
| negative-funding stretches over 3 days | 13 |
| annualised volatility of the carry stream | 0.7% |
| **carry Sharpe (funding only)** | **16.99** |

⚠ A Sharpe near 17 should be disbelieved on sight. It is computed on
the funding stream ALONE, assuming a perfect and free delta hedge.
Real basis trades carry tracking error, and the Sharpe falls fast:

| annualised hedge tracking error | resulting Sharpe |
|---|---|
| 0% | 16.99 |
| 2% | 5.60 |
| 5% | 2.35 |
| 10% | 1.18 |
| 20% | 0.59 |

## 3. ⚠ The carry is being arbitraged away

The multi-year average is dominated by 2020-21. What matters is what
is available now.

| year | annualised funding | intervals positive |
|---|---|---|
| 2020 | +17.2% | 86% |
| 2021 | +30.6% | 93% |
| 2022 | +4.2% | 78% |
| 2023 | +7.9% | 90% |
| 2024 | +11.9% | 92% |
| 2025 | +5.1% | 87% |
| 2026 | +1.9% | 67% |

**Trailing twelve months: +3.4% annualised**, positive 76% of intervals.

From 30.6% in 2021 to roughly 2-3% now. The trade was real and it has
largely been competed away. What remains does not obviously compensate
for exchange counterparty risk, margin capital and liquidation risk —
a government bond pays comparably with none of those.

This is worth stating plainly because it is the pattern of the whole
project: every edge found in this data was real at some point and has
since been arbitraged. That is what an efficient market looks like
from the inside.

