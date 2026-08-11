# ForeSight — project rules

ML and graphical analysis for cryptocurrency forecasting. Own database, own forecasting systems,
own simulation systems. **Read `README.md` for the layout, `docs/DECISIONS.md` for what has been
settled.**

The sandbox-wide git identity rules in the root `CLAUDE.md` apply here in full.

## Hard rules

1. **Never commit secrets.** Exchange API keys, wallet keys and account identifiers go in
   `.env` (gitignored), never in `config/`, never in a notebook, never in a commit message.
   `config/` holds only non-sensitive run configuration. Read-only API keys wherever the exchange
   offers them.

2. **`data/raw/` is immutable.** It is what the source returned, byte for byte, with the fetch
   timestamp recorded. Every other data stage must be reproducible from it by a committed script.
   If raw data is wrong, fix it downstream and document why — do not edit it.

3. **Point-in-time correctness is the whole game.** A crypto forecasting result that leaks the
   future is worthless, and leakage is easy to introduce silently:
   - Split chronologically. **Never** a random split of a time series.
   - Fit scalers, imputers and encoders on the training window **only**, never the whole series.
   - A feature at time *t* may use only information available at *t*. Beware resampled bars whose
     label is stamped at the open but whose values close later.
   - Exchange listings, delistings and symbol re-use are survivorship bias. Record instrument
     lifecycles in the schema rather than assuming a symbol means one thing forever.
   - `tests/` exists primarily to catch these. A leakage test is worth more than a unit test here.

4. **All timestamps are UTC, stored as such, and timezone-aware.** Never store naive local time.
   Exchange APIs disagree about milliseconds versus seconds — normalize at ingestion.

5. **A backtest that ignores costs is a fiction.** Fees, slippage and funding must be in the
   simulation engine before any strategy result is quoted. Report the no-cost number only
   alongside the with-cost one, never alone.

6. **Deterministic and script-generated.** Every figure, table and reported number comes from a
   committed script, seeded where randomness is involved. Notebooks are for exploration; once a
   result matters, it moves into `scripts/` or the relevant package.

7. **Never type a number into prose.** Reports pull their figures from the data, so they cannot
   go stale silently. *(Learned the expensive way on the SLR project.)*

## Environment

**Venvs never live in this Drive-synced folder.** Create at `~/.venvs/foresight` and reference it;
Drive sync corrupts venvs and the sandbox `.gitignore` should never have to carry one.

```
python -m venv ~/.venvs/foresight
```

## Not yet decided

Database engine, asset universe, bar interval, exchange sources. Until these are settled, do not
scaffold code that assumes any of them. Record each in `docs/DECISIONS.md` when made.
