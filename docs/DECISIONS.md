# Decision log — ForeSight

One entry per settled decision. Date it, state the choice, and state **why**, so a later session
does not re-litigate it. Open questions live at the bottom until they are answered.

---

## Settled

### 2026-08-05 — Project scaffolded
Four pillars, each owning its own layer: database (`db/` + `data/`), forecasting
(`forecasting/`), simulation (`simulation/`), and ML/graphical analysis (`analysis/`), fed by
`ingestion/` and `features/`. No dependency on any other project in the sandbox.

### 2026-08-05 — ⭐ PHASE 1 SCOPE: BTC only, and NO transfer learning (Rodrigo)

> **"I'm only interested for now in BTC — USD or USDT, and for this first part I intend to use no
> transfer learning at all. Just pure forecasting."**

**Asset universe (closes open question 2): BTC only**, quoted in USD or USDT.

**Method: single-series forecasting only.** No cross-asset transfer, no source→target framing, no
pretrained backbones. A model sees BTC's own history and nothing else.

**What this rules OUT of phase 1** — parked, not cancelled, in `BRAINSTORM.md`:
- RQ-A…RQ-E in the brainstorm are all transfer-flavoured. All deferred.
- No source-domain selection, no similarity metrics, no cross-asset experiments, no fine-tuning of
  a pretrained backbone.
- `simulation/` is not needed yet: with no strategy, there is nothing to simulate. It stays
  scaffolded and empty.

#### Amendment, same day — foundation models ARE allowed, as black boxes

> **"The FM models, if they can forecast the BTC price, I will treat them for now like a black-box
> model, and that's OK for this first part."**

**Ruling: a pretrained foundation model may be used off the shelf as a phase-1 forecaster.** It
goes in as *"a thing that takes a series and returns a forecast"*, alongside ARIMA or LightGBM.
What stays excluded is Rodrigo *doing* transfer learning: no source selection, no fine-tuning, no
adaptation, no domain-similarity work.

This is a sound call and it is cheap to act on — a zero-shot FM needs no training at all, so it
costs one inference pass to obtain a strong reference line. **If LightGBM trained on years of BTC
cannot beat a model that has never seen BTC, that is one of the most informative results phase 1
could produce.**

⚠ **Two things to keep straight, neither of which blocks anything:**

1. **Contamination changes what the number means, not whether it is valid.** Kronos is pretrained
   on OHLCV from 45+ exchanges and Chronos on a large multi-domain corpus, so BTC history may
   already be inside them. A strong "zero-shot" score could be partial memorization rather than
   generalization. **Check each model's stated pretraining corpus and record the answer beside the
   result.** It stays a legitimate benchmark either way; it just may not be a *zero-shot* one.
2. **Describe it accurately in any write-up.** "No transfer learning" would be inaccurate if a
   pretrained model is in the results table. The honest phrasing is **"no task-specific transfer;
   pretrained models used off the shelf as black-box forecasters"** — which is exactly what this
   is, and is a perfectly respectable design.

**Which FM to reach for first: Kronos.** It is pretrained on OHLCV candlesticks specifically and
consumes open/high/low/close/volume directly, which is the shape of the data being stored anyway.
Chronos and TimesFM are univariate and would need a single series chosen out of the bar. See
`PRIOR-ART.md` §2 and `MODEL-CONCEPTS.md` §4.

**What phase 1 actually is:** experiment **E1** from `BRAINSTORM.md`, the honest benchmark. One
asset, one series, engineered features, tree ensembles and neural nets, measured against
persistence on chronological splits. Nothing else.

**Why this is a good call.** The budget/transfer questions are the *interesting* ones, but every
one of them needs a trustworthy single-asset baseline to be measured against. Building that first
is the right order, and it is the phase where the leakage discipline gets established while the
system is small enough to audit.

**Consequences that follow automatically:**
- Survivorship bias stops being a phase-1 concern (one asset, still listed).
- Instrument lifecycle modelling in the schema can be deferred — but **keep the column**, since
  retrofitting it later is the painful path.
- Data volume collapses to roughly one table of a few million rows.

---

### 2026-08-05 — Source and target fixed (Rodrigo)

**Source: Binance BTCUSDT, 1-minute bars** (closes open question 5, and 3 for the stored base).
Official `data.binance.vision` archive: ~2017-present, ~4.7M bars, SHA256 checksum per file, and a
published corrections log. ⚠ Timestamps switch to **microseconds** from 2025-01-01 — normalize and
assert at ingestion.

**Target: next-bar log-return** (closes open question 4).

$$y_t \;=\; \ln\!\left(\frac{C_{t+1}}{C_t}\right) \;=\; \ln C_{t+1} - \ln C_t$$

predicted from features using **only** information available at or before $t$. The subscript is
the whole discipline: $y_t$ is stamped at $t$ but depends on $C_{t+1}$, so any feature that also
peeks at $t+1$ is leakage.

#### Why log-return and not simple return or price

| # | Property | Why it matters here |
|---|---|---|
| 1 | **Additive across time** | $\ln\frac{C_{t+2}}{C_t} = \ln\frac{C_{t+2}}{C_{t+1}} + \ln\frac{C_{t+1}}{C_t}$. A 60-minute return is the **sum** of sixty 1-minute log-returns. Simple returns must be *compounded*, $(1+r_1)(1+r_2)-1$. **This is the big one**: it makes multi-horizon targets and resampling trivial arithmetic. |
| 2 | **Stationary** | Price is non-stationary and unbounded; returns are roughly stationary. Tree models cannot extrapolate past their training range, so a non-stationary target breaks them structurally (see `MODEL-CONCEPTS.md` §1). |
| 3 | **Symmetric** | 100→110 is $+0.0953$; 110→100 is $-0.0953$. Same magnitude, opposite sign. Simple returns give $+10\%$ and $-9.09\%$ for the same round trip, and that asymmetry biases any squared-error loss. |
| 4 | **Scale-free** | A \$10 move on \$200 BTC and on \$100k BTC are not the same event. Log-returns make 2017 and 2026 comparable without rescaling. |
| 5 | **Better-behaved distribution** | Closer to symmetric and lighter-tailed than price differences. Still fat-tailed — this is crypto — but far better conditioned for MSE losses and standard tests. |
| 6 | **Numerically cheap and stable** | A subtraction of logs, not a ratio. |
| 7 | **Loses nothing at this scale** | For small $r$, $\ln(1+r)\approx r$. Minute returns are tiny (usually well under 0.1%), so log and simple returns are numerically almost identical here. You gain property 1 for free. |

#### The honest counterpoints

- **Log-returns do *not* aggregate across assets.** A portfolio's return is the weighted sum of its
  constituents' *simple* returns, not their log-returns. Irrelevant in phase 1 (one asset), but it
  will matter the moment a portfolio appears.
- **Converting back to P&L requires exponentiating.** Silently mixing log and simple returns in a
  backtest is a classic and hard-to-spot bug. Name variables `log_ret` / `simple_ret` explicitly.
- **A point mass at exactly zero.** Many 1-minute bars have no price change, so $y_t = 0$ exactly,
  far more often than any continuous distribution would predict. Worth knowing before assuming
  Gaussian anything.
- **Missing bars must never be forward-filled into fake zeros.** A gap is absence of data; a zero
  is a real observation of no movement. Conflating them invents non-events.

#### Keep the horizon configurable

$k=1$ at 1-minute is the **hardest** version of this problem and close to the efficiency limit.
Property 1 means a $k$-bar target is just a rolling sum of 1-minute log-returns, so
**parameterize $k$ from the start** and let $k \in \{1, 5, 15, 60\}$ be an experiment axis rather
than a rewrite. Signal-to-noise typically improves substantially with $k$.

---

### 2026-08-05 — Sub-minute open times are FLOORED, not dropped

**Found while building the first full interim table**, by an invariant check that failed rather
than by inspection. Binance's early kline generation sometimes stamped a bar's `open_time` a few
seconds into the minute, e.g. `2017-12-04 06:00:20.799`.

**Extent, measured against the archive:**

| Month | Misaligned bars | Of which are the ONLY bar for their minute |
|---|---|---|
| 2017-12 | 20,401 | **20,400** |
| 2018-02 | 1,201 | **1,201** |

**Ruling: floor to the minute.** These are real bars with real OHLCV, not duplicates of aligned
ones. Dropping them would have discarded **nearly half of December 2017** — the blow-off top, and
the single most informative month in the series. Exactly **one** bar collided with a properly
aligned bar for the same minute; that collision is resolved in favour of the aligned bar, which is
the canonically generated one.

**Why this was worth chasing rather than patching quietly.** Before the fix the loader reported a
**14-day "outage" in December 2017**, which is implausible on its face — that was the peak of the
mania and the most heavily traded stretch in Binance's early history. The implausibility was the
tell. The gap was an artifact of 20,400 real bars falling outside the expected minute grid, not an
exchange outage.

**Effect:** missing bars fell from 30,163 (0.64%) to **8,562 (0.18%)**, and the arithmetic now
reconciles exactly: 4,710,000 expected − 8,562 missing = **4,701,438 bars**. The remaining gaps
are all plausible Binance maintenance windows (the longest a 33-hour stop in February 2018, then
several 8–10 hour windows in 2018–2019).

---

### 2026-08-05 — Primary target switched to realized volatility (evidence-led)

**Amends the "next-bar log-return" ruling above, on evidence from
`analysis/graphical/SIGNAL-RECON.md`.** Return stays as a secondary target; it is not dropped.

| Quantity | ACF lag 1 | ACF lag 60 |
|---|---|---|
| return `r` | −0.054 | ~0.000 |
| **absolute return `\|r\|`** | **+0.418** | **+0.250** |

Direction is unpredictable past lag 3. Volatility is strongly and persistently predictable. The
lag-1 return autocorrelation *is* statistically significant at every horizon (n = 3.3M, band
±0.0011) but it is **negative** — mean reversion consistent with bid-ask bounce, living inside the
spread, and therefore not drift that survives costs.

**Primary target:** realized volatility over the next hour, on a non-overlapping hourly grid,

$$RV_t \;=\; \sqrt{\textstyle\sum_{i=1}^{60} r_{t,i}^{2}}$$

modelled in logs, which is standard practice and better behaved.

**Secondary target:** next-bar log-return, kept so the harder question stays measurable and so the
negative result is documented rather than assumed.

**Two further recon findings that shape modelling:**
- **The regime is not stable.** Zero-return share runs 25.1% (2017) → 0.1% (2021) → 4.8% (2023),
  and volatility falls ~5x from 2017 to 2023. Training across all 3.3M bars unweighted treats
  2017 Binance and 2023 Binance as one process.
- **Intraday seasonality is real** — volatility peaks 14:00 UTC, bottoms 06:00, a 1.29x spread,
  with visibly calmer weekends. Cheap and honest features.

### 2026-08-05 — Chronological split, fixed once

| Split | Range | Bars |
|---|---|---|
| train | 2017-08-17 .. 2024-01-01 | 3,343,518 |
| validation | 2024-01-01 .. 2025-07-01 | 787,680 |
| **test (held out)** | 2025-07-01 .. 2026-07-31 | 570,240 |

⛔ **The test window is not read until a model is finished being chosen.** Recon and baselines use
train and validation only. Every script that touches data states which splits it reads.

---

## Open — answer before writing code that assumes an answer

| # | Question | Status | Options / recommendation |
|---|---|---|---|
| 1 | **Which database engine?** | Open, but **easier now** | With one asset the whole dataset is ~5M rows. **DuckDB** (columnar, fast analytical scans, single file, no server) is the natural fit; SQLite also fine; Postgres/TimescaleDB is now overkill. Parquet files alone would even suffice. |
| 2 | ~~Which assets?~~ | ✅ **CLOSED 2026-08-05** | **BTC only**, USD or USDT |
| 3 | **Which bar interval?** | Open | **1m as the stored base**, resampled upward as needed. Store once at the finest granularity; every coarser bar is derivable, the reverse is not. |
| 4 | **Which forecast target?** | 🔴 **Open and now the most important question** | See the expanded note below |
| 5 | **Which exchange?** | Narrowed to a **USD vs USDT** choice | **Binance BTCUSDT** (2017+, deepest liquidity, SHA256 checksums, the reference market) *or* **Bitstamp BTC/USD** (2012+, real dollars, thin pre-2016). See recommendation below. |
| 6 | **Backtest realism baseline** | **Deferred to phase 2** | No strategy in phase 1, so no costs to model yet |

### On question 4 — the forecast target (the one that matters now)

With transfer excluded, *the target definition is the single most consequential remaining choice*.
It fixes the metric, the baseline, and whether the problem is tractable at all.

| Option | Notes |
|---|---|
| **Next-bar log-return** | The default, and what the G-Research competition used. Hardest, closest to unpredictable at 1m. |
| **k-bar-ahead return** (k = 5, 15, 60) | Aggregating over a longer horizon raises the signal-to-noise ratio. Often materially more predictable than k=1. |
| **Direction** (classification) | Easier to score and to act on, but discards magnitude, and class balance near 50% makes accuracy a misleading metric — use log-loss or AUC. |
| **Realized volatility** | **Far more predictable than returns**, with an existing benchmark literature (Optiver on Kaggle). A rational first target if returns prove hopeless. |

⚠ **Never the raw price.** Tree models cannot extrapolate beyond their training range, and
predicting price one step ahead produces a lagged copy of the input that looks perfect and
forecasts nothing. See `MODEL-CONCEPTS.md` §1 and `PRIOR-ART.md` §4.

### On question 5 — USD or USDT

**Recommendation: Binance BTCUSDT.** 2017-present is ~4.7M minute bars, which is abundant for
phase 1; it is the deepest and most-referenced BTC market; and it is the only source here shipping
per-file SHA256 checksums and a published corrections log.

**Bitstamp BTC/USD** buys history back to 2012 but that early period is thin, with many empty or
near-empty minutes and a market microstructure so different that it may actively harm a model
trained across the join. Better used as an out-of-sample robustness check on a different venue
than as the primary series.

⚠ **Do not concatenate them.** BTC/USD and BTC/USDT are different instruments with a stablecoin
basis between them.

---

## Notes carried in from elsewhere in the sandbox

- **Venvs never live in a Drive-synced folder** — `~/.venvs/foresight`. Drive sync corrupts them.
- **Never type a count or a result into prose** — generate it. This was a recurring, expensive
  defect on the SLR project; the fix is that every reported number comes from a script.
- **Chronological splits only, and fit transforms on the training window alone.** The SLR's
  risk-of-bias work found whole-series scaling before the split to be one of the most common
  leakage failures in published forecasting papers. Do not repeat it here.
