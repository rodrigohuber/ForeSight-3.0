# forecasting/

| Folder | Contents |
|---|---|
| `models/` | Model definitions only. No training loops, no data loading — a model should be importable and testable in isolation. |
| `training/` | Entry points, hyperparameter search, run tracking. |
| `evaluation/` | Metrics, baselines, statistical comparison between models. |
| `backtesting/` | Walk-forward evaluation over historical bars. |

**Every model needs a baseline it must beat.** For crypto returns the honest baselines are
persistence (next = last), a drift/mean estimate, and buy-and-hold on the asset itself. A model
reported without one is not a result.

`backtesting/` answers "would this forecast have been right"; `simulation/` answers "would acting
on it have made money". They are different questions — keep them in different folders.
