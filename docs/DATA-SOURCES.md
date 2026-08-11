# Candidate data sources — 1-minute Bitcoin

*Compiled 2026-08-05. Answers open question 5 in `DECISIONS.md` (which exchange/source to ingest
from). Each entry below was checked against its own page, not quoted from memory.*

**The single most important thing on this page:** entries 3 and 4 are **the same underlying data**
(Bitstamp BTC/USD via the Bitstamp API). Using both is not a cross-check. Real independence means
taking from **different exchanges**.

---

## Tier 1 — primary sources (the exchange publishes its own data)

These are the most trustworthy class: no intermediary, and the publisher is the venue where the
trades actually happened.

### 1. Binance — `data.binance.vision` ⭐ recommended primary
- **Repo:** <https://github.com/binance/binance-public-data> — **official** Binance repository
- **Granularity:** `1s, 1m, 3m, 5m, 15m, 30m, 1h … 1mo` — 1-minute and even 1-second
- **Scope:** spot, USD-M futures, COIN-M futures; klines, aggTrades, raw trades
- **Integrity:** every zip ships a `.CHECKSUM` (SHA256) beside it, and the repo keeps an
  *Updates* section documenting corrections with old and new checksums — the strongest data-quality
  story of anything here
- **Cadence:** daily files next day; monthly files first Monday of the month
- **Licence:** MIT (the tooling)
- **URL pattern:**
  `https://data.binance.vision/data/spot/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip`
- ⚠ **Trap:** spot data timestamps switched to **microseconds** from 2025-01-01. A parser that
  assumes milliseconds will silently mis-date everything after that date.
- ⚠ Deepest liquidity, but history starts ~2017 and it is BTC/**USDT**, not BTC/USD.

### 2. Kraken — downloadable historical OHLCVT
- **Page:** <https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data>
- **Granularity:** 1, 5, 15, 30, 60, 240, 720, 1440 minutes — includes 1-minute
- **Extra column:** **T = trade count** per bar, which most OHLCV sources drop. Useful as a
  liquidity/《thin bar》filter.
- **Scope:** from each market's inception to present (bulk file current to ~end Q3 2024)
- **Cadence:** full historical download once, then **quarterly** incremental updates
- ⚠ **Candles with no trades are omitted, not zero-filled.** Your loader must distinguish "no
  trade" from "missing data" — silently forward-filling these is a subtle way to invent liquidity
  that never existed.

### 3. Bybit — historical data download
- **Page:** <https://www.bybit.com/derivatives/en/history-data>
- Spot and perpetual klines, order book and trades. Useful as a third venue for cross-checking,
  particularly for derivatives/funding data.

---

## Tier 2 — open-source, reproducible pipelines (curated, not primary)

### 4. `ff137/bitstamp-btcusd-minute-data` ⭐ recommended for long history
- **Repo:** <https://github.com/ff137/bitstamp-btcusd-minute-data>
- **Source:** Bitstamp BTC/**USD** (a real USD pair, not a stablecoin pair)
- **Granularity:** 1-minute OHLC
- **Coverage:** from **2012-01-01**; ~6,846,600 records, ~326 MB uncompressed (~90 MB zipped)
- **Cadence:** daily, GitHub Actions at midnight UTC
- **Licence:** MIT
- **Quality claim:** "no missing minutes, no duplicates, no null values", with a **validation
  script committed in the repo** so the claim is checkable rather than asserted
- ⚠ The README states an end date of 2025-01-07 while also claiming daily updates. Probably a
  stale README line, but **verify the actual last row before trusting the coverage claim.**

### 5. Kaggle — `mczielinski/bitcoin-historical-data`
- **Dataset:** <https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data>
- **Pipeline:** <https://github.com/mczielinski/kaggle-bitcoin>
- 1-minute from January 2012, updated daily by GitHub Actions
- **The most widely cited crypto dataset in academic work** — worth using precisely because
  reviewers recognise it
- ⚠ **Source is the Bitstamp API — the same origin as entry 4.** Not an independent series.
- ⚠ The pipeline repo specifies **no licence**, so "open source" is loose here. Kaggle's own
  dataset licence governs; check it before redistributing anything derived from it.

---

## Tier 3 — aggregator

### 6. CryptoDataDownload
- <https://www.cryptodatadownload.com/data/>
- OHLCV CSVs across Binance, Bitstamp, Gemini, Bitfinex and others; daily, hourly and 1-minute.
  No login, no rate limit, no paywall.
- Convenient for a quick multi-exchange comparison. But it is a **re-publisher**: you inherit
  whatever processing it applied, and you cannot verify against a checksum the way you can with
  Binance. Fine for exploration, weak as a system of record.

---

## Recommendation for ForeSight

| Role | Source | Why |
|---|---|---|
| **Primary** | Binance | Checksums, corrections log, deepest liquidity, spot **and** futures, 1s if ever needed |
| **Long history** | Bitstamp via `ff137` | Reaches back to 2012, real USD pair, MIT, self-validating |
| **Cross-check** | Kraken | Independent venue, and the trade-count column exposes thin bars the others hide |

Three venues, three genuinely independent price series. Disagreement between them at a given
minute is **information** — it flags outages, bad prints and thin liquidity — which is exactly
what a single-source dataset cannot tell you.

## Caveats that apply to all of them

1. **"Open source" describes the tooling and the licence, not always the data.** Exchange terms of
   use may restrict redistribution even when download is free. Check before republishing anything
   derived.
2. **BTC/USD ≠ BTC/USDT.** Bitstamp and Kraken quote real dollars; Binance's deep pair is USDT.
   These are different instruments with a stablecoin basis between them. Never concatenate them
   into one series.
3. **Early Bitstamp history (2012–2016) is thin.** Many minutes have little or no volume. A
   "complete, no missing minutes" dataset covering that era is filling gaps by construction —
   understand how before modelling on it.
4. **Timestamp units differ and change over time** (see the Binance 2025 microsecond switch).
   Normalize to UTC at ingestion and assert the result, per rule 4 in `CLAUDE.md`.
