<div align="center">

# ForeSight 3.0

### Forecasting you can audit.

**A Web3 forecasting platform combining neural networks, transfer learning and explainable AI with
blockchain infrastructure — built to deliver predictions that are auditable and
manipulation-resistant through on-chain oracles.**

*Zero-shot and few-shot transfer learning to generalise to new assets with little history ·
SHAP explainability so every signal can answer "why" · built for DeFi protocols and DAOs that
demand methodological transparency.*

</div>

---

## Everyone shows you the backtest that worked.

Here is ours, and the six that died.

That is the entire pitch. In a market where every forecasting product arrives with a beautiful
equity curve and no way to check it, ForeSight 3.0 leads with its own graveyard — because a method
that has never killed one of its own ideas has never been tested.

| we tested | verdict | the evidence |
|---|---|---|
| Minute-scale direction from price | 💀 **DEAD** | R² −1.02 against predicting zero. 49.4% hit rate. |
| The 38 classic chart indicators | 💀 **DEAD** | 1 survives multiple-testing correction — and points the wrong way. |
| GARCH / GJR-GARCH / EGARCH | 💀 **BEATEN** | all three land *below* a naive carry-forward. |
| Our vol model vs. the options market | 💀 **LOST** | Deribit DVOL RMSE 0.227 against our 0.311. |
| Variance risk premium, after costs | ⚖️ **BREAKEVEN** | breakeven at ~1.5 vol points round-trip. |
| Funding carry (long spot / short perp) | ⏳ **ARBITRAGED** | 30.6% in 2021 → 3.4% trailing twelve months. |
| Positioning data (OI, top-trader ratios) | 💀 **DEAD** | 0 of 16 features survive correction. |
| **Trend following + vol-scaled sizing** | ✅ **SURVIVES** | **beats buy-and-hold on BTC *and* ETH.** |

**Seven tested. Six dead. One survives.** Anyone can show you the last row. We publish all eight.

---

## The one that survived — and it survived twice

Long when the trailing return is positive, sized inversely to recent volatility, otherwise flat.
Daily bars, 10 bps of costs. **The full parameter grid, not its best cell:**

| lookback | BTC Sharpe | ETH Sharpe | BTC max drawdown |
|---|---|---|---|
| **30d** | **0.99** | **0.81** | **0.54** |
| 60d | 0.92 | 0.73 | 0.72 |
| 90d | 0.61 | 0.49 | 0.71 |
| 180d | 0.59 | 0.31 | 0.73 |
| 365d | 0.61 | 0.40 | 0.66 |
| *buy & hold* | *0.45* | *0.25* | *1.15* |

> **Every lookback beats buy-and-hold, on both assets, at every cost level up to 20 bps, with
> drawdowns roughly halved.**

**Why the ETH column is the important one.** The rank ordering across lookbacks is nearly identical
on two independent assets. A pattern that exists only on Bitcoin is a Bitcoin fact, not a method —
so the replication is what turns a backtest into a finding. The value concentrates in **downside
protection**: through the 2021–22 drawdown, every variant lost less than holding.

**And the limits, stated by us before a referee states them for us:** Sharpe standard error is
0.35, so the margin over buy-and-hold is 0.5–1.5 SE — suggestive, not conclusive. Both assets rose
enormously across the window, which flatters a long-biased rule. Trend following is a crowded,
documented risk premium, not our discovery. A backtest without live slippage is not a live result.

*Any product that cannot write that paragraph about itself is hiding it.*

---

## 💡 The finding that cost the most to learn

### A better forecast is a **worse** position sizer.

| sizing rule | strategy Sharpe | turnover / yr |
|---|---|---|
| simple 20-day realized vol | **0.61** | **7×** |
| our LightGBM vol forecast | 0.50 | 48× |

Our LightGBM model beats HAR by **30% on forecast accuracy** — and loses as a sizer, at **7× the
turnover**. For an always-long book it is worse still: 0.07 against 0.45.

**Forecasting rewards responsiveness. Sizing rewards stability. They are different objectives, and
no amount of additional accuracy fixes the gap.**

This is what the academic-to-operational bridge is actually worth. A team optimising forecast
error will spend months on the wrong objective and never see it in their metrics. We already paid
that bill, and we measured it.

---

## 📉 We watched an entire edge die in public

Annualised funding rate, Binance, 7,212 settlements:

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|
| +17.2% | **+30.6%** | +4.2% | +7.9% | +11.9% | +5.1% | +1.9% |

Trailing twelve months: **+3.4%**. A government bond pays comparably, without exchange counterparty
risk, margin capital, or liquidation risk. The carry trade was real, and it has been arbitraged
away.

The naive carry computes to **Sharpe 17** — a number that should be disbelieved on sight, since it
assumes a perfect free delta hedge. At 5% hedge tracking error it is 2.35; at 10%, 1.18. **We
report the number and the reason not to believe it in the same breath.** That habit is the product.

---

## What runs today

| | |
|---|---|
| **Own data infrastructure** | BTC + ETH 1-minute bars, 2017–2026 — **9.4M bars**, every source archive SHA-256 verified |
| **Derivatives & sentiment** | Deribit DVOL (32,987 hourly), Binance funding (7,212 settlements), positioning metrics (377,719 points) |
| **Backtest engine** | costs charged everywhere, walk-forward, chronological splits, leakage tests |
| **A validated strategy** | trend following, replicated across two independent assets |
| **Live forecasting** | `scripts/forecast_now.py` — forecast, log, and score against what actually happened |
| **Reproducibility** | 27 Python modules, every reported number generated by a committed script |

## Where it is going

- **Live paper trading** — the real next step. The trend result needs contact with actual slippage
  far more than it needs another parameter sweep.
- **On-chain publication via oracle** — signals written to chain, timestamped and
  manipulation-resistant, so a DAO can audit a forecast instead of trusting one.
- **Per-signal SHAP explainability** — every prediction answers *why*, which is what a treasury
  committee actually asks.
- **Zero-shot / few-shot transfer learning** — generalising to assets with little history. This is
  the core of the founder's doctoral research in forecasting and time series, transferred across
  domains rather than invented for a pitch deck.
- **Automatic retraining on concept-drift detection** — forecasts that stay calibrated as the
  regime moves.

---

## 🔬 The product is not the signal. It is the process that rejects signals.

Every result above was produced under these rules, and they are enforced in the code, not promised
in a deck:

1. **Chronological splits only** — never a random split of a time series. Scalers, imputers and
   encoders fit on the training window alone.
2. **Walk-forward evaluation** — every block is scored *before* it is trained on. No lookahead bias
   inflating the history.
3. **Costs charged everywhere.** A backtest that ignores costs is fiction; the no-cost number is
   never reported alone.
4. **Multiple-testing correction** (Benjamini-Hochberg) when testing 38 indicators at once — which
   is exactly why 37 of them died.
5. **The full parameter grid is reported**, never its best cell. That is the difference between a
   result and an advertisement.
6. **Point-in-time correctness is the whole game.** `tests/` exists primarily to catch leakage; a
   leakage test is worth more than a unit test here.
7. **Provenance** — every source archive SHA-256 verified against its published checksum, every
   reported number generated by a committed script, never typed into prose.

### Two bugs we caught, and are telling you about

Both produced believable *wrong* answers, and both were caught by the results being **implausible**
rather than by a test failing. How a project catches its own errors is part of what its numbers are
worth:

- **A December-2017 "14-day outage"** that never happened — 20,400 real bars falling outside the
  expected minute grid. The tell was implausibility: that fortnight was the blow-off top and the
  most heavily traded stretch in Binance's early history. Full ruling in
  [`docs/DECISIONS.md`](docs/DECISIONS.md).
- **A forward-return leak.** The first momentum signal was built from the `cumsum` of *forward*
  returns, so the running total at *t* already contained the return being predicted — correlation
  exactly **1.000**. Rebuilt from trailing returns; the conclusion survived, but it had to be
  *measured*, not assumed.

---

## Who this is for

**DeFi protocols · DAOs · treasuries · desks** — anyone who has to *defend* an allocation decision
in public, to a community or a risk committee, rather than merely make one.

If your governance forum will ask "where does this number come from and what else did you try?",
this repository is built to answer both.

---

## Data

Not in this repository — it is large and partly redistribution-restricted. The code expects it at
`FORESIGHT_DATA_ROOT`. See [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md) for exact endpoints and
the fetch procedure.

## Layout

```
├── config/              Run configuration. NO secrets.
├── db/                  Schema, forward-only migrations, shared queries
├── data/                raw (immutable) → interim → processed; external reference data
├── ingestion/           Per-exchange collectors
├── features/            Feature engineering, one module per family
├── forecasting/         models · training · evaluation · backtesting
├── simulation/          engine · strategies · scenarios
├── analysis/            graphical · notebooks · reports
├── scripts/             Deterministic entry points: fetch, build, train, simulate, report
├── tests/               Especially leakage and point-in-time correctness tests
└── docs/                Design notes, data dictionary, decision log
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then set FORESIGHT_DATA_ROOT and any read-only API keys
```

Never commit `.env`. Exchange API keys, wallet keys and account identifiers belong there and
nowhere else; use read-only keys wherever the exchange offers them.

Full method detail: [`PHASE1-REPORT.md`](PHASE1-REPORT.md), [`METHOD.md`](METHOD.md), and the nine
script-generated result files in [`forecasting/evaluation/`](forecasting/evaluation/).

---

<div align="center">

**Tested, retested, and reported in full — data science, not hype.**

Phases 1 and 2 complete. This repository is research and is not investment advice.

</div>
