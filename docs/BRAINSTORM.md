# ForeSight — brainstorm and idea bank

*Session of 2026-08-05. Everything discussed, captured so nothing is lost. Nothing here is
committed to; `DECISIONS.md` is where things become real.*

**This file is the index.** The detail lives in four siblings:

| Doc | What is in it |
|---|---|
| [`DECISIONS.md`](DECISIONS.md) | Settled choices + the 6 blocking open questions |
| [`DATA-SOURCES.md`](DATA-SOURCES.md) | 6 verified 1-minute BTC sources, ranked by trust |
| [`MODEL-CONCEPTS.md`](MODEL-CONCEPTS.md) | The model families and why each is on the list |
| [`PRIOR-ART.md`](PRIOR-ART.md) | What already exists, what not to rebuild, positioning |

---

## 1. The core thesis, in one paragraph

Everyone in crypto forecasting optimises **which model** and **which features**. Almost nobody asks
**how much of an asset's own history a model actually needs before it beats a model that has never
seen that asset at all.** That question — the **target-data budget** as the organizing variable —
is the differentiator of Rodrigo's SLR, and it transfers to crypto with unusual force because
crypto has a *permanent* cold-start problem: new pairs list constantly, so zero-history forecasting
is routine here rather than rare.

**ForeSight's angle is therefore not "predict Bitcoin". It is "measure the price of ignorance":
how much local history is worth how much forecast skill, and when does a frozen foundation model
beat training on your own data.**

---

## 2. Research questions worth testing

Ordered by how much they build on work already done.

### RQ-A — Does the SLR's budget map survive the move to crypto?
The SLR's cross-study map (solar, 105 studies) says:

| Target budget | Best strategy |
|---|---|
| 0 to ~2–4 weeks | zero-shot / frozen transfer; adaptation often *hurts* |
| ~1 to 13 weeks | lightweight adaptation only (linear heads, adapters, state transfer) |
| ≳ 90 days | full fine-tuning finally pays |
| ~4–12 months | local-only training becomes competitive; transfer's residual value is compute |

**Does this hold in a market with 24/7 trading, no closing auction, no overnight gap, and much
higher volatility?** Crypto has no market close, which removes one of the strongest seasonal
structures solar forecasting relies on. The boundaries could shift in either direction, and either
result is interesting.

### RQ-B — Where is the crossover for a newly listed pair?
For a pair listed on day 0: plot forecast skill against days of its own history, with a zero-shot
foundation model as a flat reference line. **The crossover point is the deliverable.** This is the
crypto analogue of Figure 2-B in the SLR.

### RQ-C — Does domain-pretraining beat generic pretraining?
The SLR's `FM` cluster splits into text-pretrained (5) and generic-time-series-pretrained (3).
**Kronos adds a third kind: pretrained on OHLCV candlesticks specifically.** Ranking those three
against each other on the same crypto task would be a clean, novel contribution — and it is a
direct extension of a table that already exists in the manuscript.

### RQ-D — Is there a similarity metric that predicts transfer success between assets?
The SLR's RQ3 finding: **65 of 105 studies use no similarity metric at all**, and only **two**
validate their metric against realized transfer outcome. In crypto the question becomes: does
BTC→ETH transfer better than BTC→some illiquid altcoin, and does any measurable similarity
(correlation, DTW, distribution distance, market-cap tier, liquidity) predict it? **This is the gap
the SLR names and the corpus does not fill.** Mendes et al. 2025 already did exactly this for solar
sites across Brazil — the method is transferable.

### RQ-E — How much does the answer change with bar interval?
Does the budget curve look different at 1m, 1h and 1d? A model may need 30 days of daily bars but
2 days of minute bars for the same number of observations — **or the reverse, if minute-scale noise
means observations are worth less.** Nobody appears to have separated "number of observations" from
"span of calendar time" in this literature.

---

## 3. Experiment designs

### E1 — The honest benchmark (must come first)
Fixed target: next-bar **log-return**. Fixed split: chronological, walk-forward. Fixed baselines:
**persistence, drift, buy-and-hold**. Then run, in order of cost: LightGBM → XGBoost → Random
Forest → MLP → TCN/Transformer → zero-shot TSFM.
**Deliverable:** a table saying which, if any, beats persistence, and by how much, with dispersion
across seeds. Expect the honest answer to be "barely, and only sometimes". That is a result.

### E2 — The budget curve (the actual contribution)
Take N assets. For each, train at budgets of 1, 3, 7, 14, 30, 60, 90, 180, 365 days of its own
history. Evaluate all on the same held-out future window. Overlay the zero-shot foundation model as
a horizontal line. **The x-axis where the curves cross is the answer to RQ-B.**
*This is deliberately the same day-grid the SLR's §10 proposes as a prospective protocol —
so the review's own recommendation gets executed in a second domain.*

### E3 — The cold-start replay
Find pairs that listed recently. Replay each from its listing date forward, forecasting from day 1
with (a) zero-shot FM, (b) transfer from a related asset, (c) local training as data accrues.
**This is the most realistic version of the question and the most operationally useful.**

### E4 — Cross-asset transfer matrix
All-pairs source→target transfer over a basket, then correlate transfer gain against candidate
similarity metrics. Answers RQ-D and directly extends the Mendes et al. 2025 method.

---

## 4. Ideas parked (raised, not developed)

- **On-chain features as a second domain.** `ingestion/onchain/` exists for this. Cross-variable
  transfer (on-chain → price) would mirror the SLR's `cross-variable` transfer direction, which had
  7 instances in the solar corpus.
- **Regime conditioning.** The SLR found similarity-to-budget coupling in exactly one study (Min &
  Noh set the fine-tuning ratio by similarity tier). The crypto analogue: set the retraining budget
  by detected volatility regime.
- **Cross-resolution transfer.** Suemitsu 2026 is the SLR's only cross-frequency study — a model
  deployed zero-shot at time resolutions it was never trained on. Crypto has clean nested
  resolutions (1m/5m/1h/1d) from a single source, making this much easier to study here than in
  solar.
- **Negative transfer prevalence.** The SLR says no study estimates it and no benchmark exposes it.
  A crypto all-pairs matrix would measure it directly — arguably the cleanest way to answer a
  question the SLR explicitly leaves open.
- **Volatility rather than return as the target.** Better-behaved, more predictable, and there is
  an existing literature (Optiver Realized Volatility on Kaggle) to benchmark against. A safer
  first target if returns prove hopeless.
- **Ensemble across foundation models.** Chronos + TimesFM + Moirai + Kronos as a committee, since
  each has a different pretraining corpus.

---

## 5. Publication angles

1. **The strongest:** "How much local data does a crypto forecasting model need?" — a direct
   companion paper to the SLR, in a new domain, executing the protocol the SLR itself proposes.
2. **The safest:** a benchmark/reproducibility paper. Honest baselines, chronological splits,
   costs included — the field is full of results that would not survive that treatment.
3. **The most novel:** domain-pretrained vs generic-pretrained vs text-pretrained foundation models
   (RQ-C), which extends a table already in the manuscript.
4. **A methods note:** the leakage taxonomy from the SLR's risk-of-bias work, applied to crypto ML,
   with a test suite. Small but genuinely useful, and `tests/` would be the artifact.

---

## 6. Hard truths to keep in view

- **Minute-scale crypto returns are close to unpredictable.** G-Research winners achieved small
  positive correlations, not accuracy. Any high R² means leakage.
- **The "LSTM predicts Bitcoin" genre is largely an artifact** — predicting price one step ahead
  produces a lagged copy of the input that looks perfect and forecasts nothing. See `PRIOR-ART.md`
  §4.
- **Tree models cannot extrapolate.** Model returns, never raw price. See `MODEL-CONCEPTS.md` §1.
- **Foundation-model contamination is real.** Kronos is pretrained on 45+ exchanges; Chronos on a
  large multi-domain corpus. "Zero-shot on BTC" may not be zero-shot. Check each model's stated
  corpus before quoting.
- **Costs decide everything.** A strategy that is profitable before fees and slippage is not a
  finding. Rule 5 in `METHOD.md`.
- **Survivorship bias is severe in crypto.** Dead coins vanish from most datasets. A backtest over
  today's listed pairs is a backtest over the winners.

---

## 7. Assets already owned that carry over

- **The SLR's budget map** — a published hypothesis to test, not a guess.
- **The SLR's risk-of-bias taxonomy** — the leakage failure modes are already enumerated and were
  observed empirically across 105 papers.
- **Mendes et al. 2025's similarity method** — transfer loss correlated against four similarity
  metrics across twenty sites. Directly reusable for RQ-D.
- **The reproducibility discipline** — generated numbers, committed scripts, no typed counts.
  Learned expensively on the SLR; `METHOD.md` carries it forward.
- **The extraction-schema habit** — freeze the experiment schema before running, not during.

---

## 8. Immediate next actions

1. Answer the 6 blocking questions in [`DECISIONS.md`](DECISIONS.md) — database, assets, interval,
   target, exchange, cost baseline. **Nothing should be coded before these.**
2. Read the G-Research top solution write-ups before designing `features/`.
3. Check **Kronos's pretraining corpus** for crypto contamination — it determines whether it can
   serve as a zero-shot baseline at all.
4. Build E1, the honest benchmark, and let it decide whether E2 is worth running.
