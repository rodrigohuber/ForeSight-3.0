# data/

Four stages, one direction: `raw -> interim -> processed`. `external/` sits beside them.

| Stage | Rule |
|---|---|
| `raw/` | **Immutable.** Exactly what the source returned, with the fetch timestamp recorded. Never edited. |
| `interim/` | Cleaned, aligned, resampled, gap-handled. Fully reproducible from `raw/` by a committed script. |
| `processed/` | Model-ready feature matrices and label sets. Reproducible from `interim/`. |
| `external/` | Third-party reference data: calendars, macro series, symbol mappings. |

Nothing in here is versioned in git (see `.gitignore`) — the scripts that build it are, which is
what makes the pipeline reproducible rather than the files.

**Point-in-time discipline:** a row stamped at time *t* may contain only what was knowable at *t*.
If you cannot say when a value became known, it does not belong in `processed/`.
