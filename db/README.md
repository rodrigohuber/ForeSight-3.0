# db/

The database definition. The data itself lives in `data/`, and database files are gitignored.

| Folder | Contents |
|---|---|
| `schema/` | DDL. Expected core tables: instruments (with listing/delisting lifecycle), OHLCV bars, features, model runs, forecasts. |
| `migrations/` | Ordered, forward-only. Never edit a migration that has been applied — add another. |
| `queries/` | Reusable SQL needed by more than one component, so the same definition is not re-typed in three places. |

**Engine not yet chosen** — see `docs/DECISIONS.md` open question 1. Do not write schema that
assumes one until it is settled.

Two things the schema must handle from the start, because retrofitting them is painful:
- **Instrument lifecycles.** Symbols get listed, delisted, and re-used. A symbol is not an identity.
- **UTC everywhere**, timezone-aware, with the source's original timestamp preserved alongside the
  normalized one.
