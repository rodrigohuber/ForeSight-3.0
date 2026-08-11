# Model concepts for price forecasting — ForeSight

*Rodrigo's working shortlist, 2026-08-05. A thinking document, not a commitment: nothing here is
chosen yet. Kept alongside `DECISIONS.md`, which records what actually gets settled.*

The families below run from cheapest to most expensive, and that order is deliberate — each one
should have to beat the one above it before it earns its place.

---

## 1. Tree ensembles — Random Forest, XGBoost, LightGBM

**What they are.** Ensembles of decision trees over a tabular feature matrix. Random Forest grows
trees independently and averages them (bagging); XGBoost and LightGBM grow them sequentially, each
correcting the last one's errors (boosting). LightGBM is the fastest of the three on wide feature
sets — it grows leaf-wise and bins features into histograms.

**Why they belong on the list.** On tabular problems with engineered features they are still the
strongest default, they train in seconds, they need almost no tuning to be competitive, and they
give you feature importances that are actually interpretable. For a first honest benchmark on
crypto they are the right starting point.

> ### ⚠ The one thing that breaks tree models on prices
> **A tree cannot extrapolate.** Its prediction is always an average of training-set leaf values,
> so it can never output a number outside the range it was trained on. Feed it raw BTC price and
> the moment price exceeds its training maximum, every forecast pins to the ceiling — and it will
> look excellent in backtest right up until it doesn't.
>
> **Therefore: model returns, log-returns, or another stationary transform. Never raw price.**
> This is not a tuning preference, it is a structural limit of the model class.

**Other crypto-specific cautions.**
- They see no time axis. Lags, rolling statistics and time-of-day features must be built by hand
  in `features/` — the model will not discover them.
- They will happily learn from a leaked feature. Point-in-time discipline matters more here than
  model choice.
- Boosting overfits noisy financial targets fast. Early stopping on a **chronologically later**
  validation split, never a random one.

---

## 2. MLP — the plain feed-forward network

**What it is.** Fully-connected layers on the same tabular feature matrix the trees consume.

**Why keep it.** It is the honest control for "does deep learning buy me anything here?" On
tabular financial data an MLP frequently *loses* to LightGBM, and that result is worth having
explicitly rather than assumed. It also extrapolates where trees cannot, which makes the
comparison with §1 informative rather than merely competitive.

**Cautions.** Needs feature scaling (fit on the training window only). Sensitive to
initialisation, so report across seeds with dispersion, not a single lucky run.

---

## 3. Deep sequence networks — LSTM/GRU, TCN, Transformer

**What they are.** Models that consume the sequence directly instead of a hand-built feature row.
LSTM/GRU carry a recurrent state; TCNs use dilated causal convolutions; Transformers use
self-attention over the window.

**Why they belong.** They learn temporal structure you did not think to engineer, handle
multivariate inputs naturally (price + volume + order-book + on-chain), and extend to multi-horizon
output.

**Cautions.**
- Data-hungry and slow relative to §1. Minute bars help here — there are millions of them.
- **Causal masking is not optional.** A non-causal convolution or an unmasked attention window
  reads the future. This is leakage that a metric will never reveal, because the model looks
  brilliant.
- Easy to beat a weak baseline and call it success. Persistence (next = last) is remarkably hard
  to beat on minute-scale returns — measure against it every time.

> **Cross-reference from the SLR:** across the 105 included studies, backbones split roughly
> hybrid 50 · RNN/LSTM/GRU 24 · Transformer 14 · MLP 10 · **tree-ensemble only 4** · CNN 4. The
> solar TL literature is overwhelmingly neural. Your §1 shortlist is the *rare* choice there —
> which is a point in its favour for a benchmark, not against it.

---

## 4. Time-series foundation models — the SLR's `FM` cluster

**This is the family you were reaching for, and the terminology is worth getting exactly right,
because you were not simply wrong about it.**

### The naming question, settled

Chronos and Lag-Llama **are** language-model architectures. Chronos quantises a series into a token
vocabulary and trains a **T5**; its paper is literally titled *"Learning the Language of Time
Series"*. Lag-Llama uses the **LLaMA** decoder, as the name says. When you pushed back on the claim
that these "are not language models", you were right and the record was corrected.

So "LLM" is **architecturally defensible**. What makes it imprecise is that the single word is
doing two jobs at once:

| Axis | Question it answers |
|---|---|
| **Architecture** | Is the backbone a transformer from the LLM family? |
| **Pretraining source** | Was it pretrained on **text**, or on **numeric time series**? |

For a review about *what is transferred and from where*, the second axis is the consequential one —
and it is the one the word "LLM" hides.

### What your own cluster actually contains

The `FM` cluster is **9 of 105 studies, all published 2025-26**, and `extraction.csv`'s
`source_domain` field already splits it:

| Pretrained on | n | Studies |
|---|---|---|
| **Text** | 5 | Pegasus/C4 (He 2026) · GPT-2 (Song 2026; Sun 2025) · frozen Llama-7B (Liu 2026) · DeepSeek-R1 14B (Chen 2026) |
| **Time series** | 3 | Chronos, ~890k series (Wenlong 2025; Shan 2026) · Lag-LLaMA (Zhang 2026) |
| **Neither** | 1 | Suemitsu 2026 — trained from scratch, only *benchmarked against* a foundation model |

**The reportable finding:** the literature itself calls TSFMs "LLMs". Wenlong 2025 is titled
*"Short-Term Solar Power Forecasting with **Large Language Model**"* and runs Chronos; Zhang 2026
says *"**large language model**-based approach"* and runs Lag-LLaMA. Only Shan 2026 names it
correctly as a *time-series foundation model*. So the imprecision you noticed is **the field's, not
yours** — and it is now a sentence in the manuscript.

### What this means for ForeSight

The interesting family for crypto is the **time-series-pretrained** one — Chronos, Lag-Llama,
Moirai, TimesFM, TimeGPT. These are pretrained on large multi-domain corpora of numeric series and
can forecast a series they have never seen, **zero-shot**, with no training on your data at all.

**Why that is worth testing here.**
- A genuine zero-target-data baseline: a number you can obtain on day one, before any training.
- Your SLR's central finding is that below roughly 2–4 weeks of target data, frozen transfer beats
  adaptation. A newly listed crypto pair is exactly that cold-start situation.
- It reframes the project's core question from "what model" to "how much of my own history do I
  actually need" — which is the same question your PhD is already about, in a different market.

**Cautions.**
- **Contamination is a real risk.** These models were pretrained on large public corpora that may
  already include crypto price history. A "zero-shot" result on BTC may not be zero-shot at all.
  Check each model's stated pretraining corpus before quoting a result.
- Text-pretrained backbones (the 5 above) are a separate and much weaker claim. Do not conflate
  them with the time-series-pretrained 3 — that conflation is the exact problem this section exists
  to name.
- Heavy at inference relative to LightGBM. Cost belongs in the comparison.

---

## Suggested order of work

1. **Baselines first** — persistence, drift, buy-and-hold. Nothing counts until these are beaten.
2. **LightGBM on engineered features, target = log-returns.** Fast, strong, and it exposes whether
   the features carry signal at all.
3. **MLP on the same matrix** — the "is deep learning buying anything" control.
4. **One sequence model** (TCN or a small Transformer) if 2–3 show the features are limiting.
5. **A time-series foundation model, zero-shot**, as the cold-start anchor and the bridge to the
   SLR's data-budget question.

Every step reports against the same baseline set, on chronological splits, with costs included
once a strategy is attached. See rules 3, 5 and 6 in `CLAUDE.md`.
