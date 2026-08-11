# Prior art and positioning — ForeSight

*Compiled 2026-08-05. What already exists, what not to rebuild, and where the actual gap is.*

---

## 1. The closest direct precedent: G-Research Crypto Forecasting (Kaggle)

<https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/>

**This is essentially ForeSight as a competition.** Forecast short-term returns of **14 crypto
assets** from **minute-level** data, scored on **weighted Pearson correlation**. 1,946 teams,
$125,000 prize pool.

Why it matters more than any repo on this page:

- It settles the **target design** question. The winners forecast *short-horizon returns*, not
  price, and were scored on correlation rather than RMSE — because on minute data the achievable
  signal is a tiny correlation, not an accurate price.
- The public write-ups are a free, adversarially-tested feature library. One documented top-1%
  finish used **LightGBM with no hyperparameter tuning**, and attributed the result to three
  things only: **feature construction, training-data resampling, and feature selection.**
- It calibrates expectations honestly. Top scores were small positive correlations. Anyone
  reporting high R² on minute-level crypto returns has leaked something.

**Action:** read the top solution write-ups before designing `features/`. Do not re-derive what
1,946 teams already searched.

---

## 2. ⭐ Kronos — a foundation model for OHLCV candlesticks

<https://github.com/shiyu-coder/Kronos> · MIT · AAAI 2026 · Shi, Fu, Chen, Zhao, Xu, Zhang & Li (2025)

**This is the single most relevant find for you, because it is your SLR's `FM` cluster idea built
directly for financial data.**

- **First open-source foundation model for financial candlesticks (K-lines)**, pretrained on OHLCV
  from **45+ global exchanges**.
- Two-stage design: a **tokenizer quantizes multi-dimensional OHLCV into hierarchical discrete
  tokens**, then a **decoder-only autoregressive Transformer** is pretrained on those tokens.
- Models on Hugging Face (`NeoQuasar/Kronos-*`); fine-tuning scripts released 2025-08-17.

**Why this matters to your framing.** It is the *same architectural move as Chronos* — quantize a
numeric series into a token vocabulary, then train a language-model architecture on it — but with
the pretraining corpus being **market data rather than generic time series or text**. On your
SLR's own axis it is a **fourth category**: not text-pretrained, not generic-time-series-pretrained,
but **domain-pretrained on exactly the data type you are forecasting**.

That makes it both the strongest baseline available and a live contamination risk: if its
pretraining corpus includes crypto exchanges, a "zero-shot" BTC result is not zero-shot. **Check
its stated corpus before quoting any number from it.**

---

## 3. Full platforms — do not rebuild these

| Project | What it is | Take from it |
|---|---|---|
| **[Freqtrade](https://github.com/freqtrade/freqtrade) + FreqAI** | ~25k★, crypto-native, since 2017. Backtesting, hyperopt, live execution. FreqAI does **self-adaptive retraining** in live deployment, feature sets of 10k+, and RL. | The closest thing to ForeSight that already exists. Its `feature_engineering_*` convention and its retraining loop are worth copying wholesale. |
| **[Microsoft Qlib](https://github.com/microsoft/qlib)** | AI-oriented quant platform from MSRA. Full pipeline: data → model → backtest → portfolio → execution. Supervised, market-dynamics and RL paradigms; explicit high-frequency support. | The **architecture reference** for how to separate data, model, strategy, backtest and execution into loose-coupled modules. Its layout validates the one in `README.md`. |
| **[NautilusTrader](https://github.com/nautechsystems/nautilus_trader)** | Event-driven trading systems framework, Rust core. | The realistic-execution end: the one engine of the three that gives a fill price you can trust. Reference for `simulation/engine/`. |
| **[VectorBT](https://github.com/polakowo/vectorbt)** | Vectorized backtesting; thousands of parameter combinations in seconds. | Fast sweeps and robustness testing. ⚠ Weak on partial fills, slippage and queueing — research tool, not a P&L oracle. |
| **[CCXT](https://github.com/ccxt/ccxt)** | Unified API across 100+ exchanges. | `ingestion/exchanges/` should wrap this rather than write per-exchange REST clients. |
| **[awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading)** | Curated index of libraries, strategies, books. | The map. Check it before writing anything from scratch. |

---

## 4. ⚠ The failure mode most crypto-prediction repos share

Search GitHub for "bitcoin price prediction LSTM" and you get hundreds of repositories showing a
chart where the predicted line tracks the actual line almost perfectly.

**Almost all of them are predicting price rather than return, one step ahead.** The model learns
`price(t+1) ≈ price(t)`, and the beautiful chart is a lagged copy of the input. It is persistence
with extra steps, and it has no predictive value whatsoever. The tell is a visibly high R² and a
prediction curve that appears shifted one bar to the right.

`CryptoPredictions` and similar comparison libraries are useful as *implementation* references
(ARIMA/Prophet/RF/XGBoost/LSTM side by side), but treat any headline accuracy from this genre as
unverified until you have seen the target definition and the split.

**Guard in ForeSight:** every model reports against persistence, on returns, with a chronological
split. That single rule immunises the project against this entire category of result.

---

## 5. Where the gap actually is — the positioning

Everything above optimises **which model** and **which features**. Almost nothing asks:

> **How much of an asset's own history does a forecasting model actually need?**

That is your SLR's differentiator — the **target-data budget** as the organizing variable — and it
transfers to crypto with unusual force:

- **Crypto has a permanent cold-start problem.** New pairs list constantly. Every listing is a
  zero-history forecasting problem, which in equities is rare and here is routine.
- **The zero-shot machinery now exists** — Chronos, TimesFM, Moirai, Lag-Llama, and Kronos
  specifically for OHLCV. The question "frozen foundation model versus a model trained on this
  asset's own 30 days" is newly answerable and, as far as this search found, not systematically
  answered for crypto.
- **Your SLR already produced a hypothesis to test.** Its cross-study map says: below roughly 2–4
  weeks of target data, frozen transfer wins and adaptation often hurts; lightweight adaptation
  wins from about one to thirteen weeks; full fine-tuning pays only beyond ~90 days. **Does that
  map hold in a market with 24/7 trading, no closing auction, and far higher volatility?**

That is a genuine research question, it is publishable, and it reuses the analytical apparatus you
have already built rather than starting a second unrelated project.

---

## 6. Three candidate project shapes

| Shape | The question | Effort | Novelty |
|---|---|---|---|
| **A — Budget curve for crypto** | For a newly listed pair, how does forecast skill improve with days of its own history, and where does a zero-shot foundation model stop being the better choice? | Medium | **High** — directly extends the SLR into a new domain |
| **B — Honest benchmark** | Do LightGBM/XGBoost/RF, an MLP, a sequence model and a zero-shot TSFM actually beat persistence on minute-scale crypto returns, under identical splits and costs? | Low–medium | Moderate, but it is the necessary foundation for A |
| **C — Trading system** | Turn forecasts into a strategy with realistic fills, fees and slippage. | High | Low as research; high as a personal tool |

**Recommended order: B → A → C.** B builds the infrastructure and the baselines that A needs, and
without B's baselines A's curve would be measuring nothing. C should not begin until A says there
is something worth trading.

---

## 7. What to reuse rather than write

- **Exchange connectivity** → CCXT, not hand-rolled REST clients.
- **Technical indicators** → `pandas-ta` or TA-Lib, not hand-rolled RSI.
- **Vectorized sweeps** → VectorBT.
- **Execution realism** → study NautilusTrader's fill model before writing `simulation/engine/`.
- **Feature conventions** → FreqAI's `feature_engineering_*` pattern.
- **Pipeline architecture** → Qlib's module separation.

**Write yourself:** the budget-curve experiment design, the point-in-time feature store, the
leakage tests, and the evaluation protocol. Those are where the contribution is, and they are
exactly what no existing project hands you.
