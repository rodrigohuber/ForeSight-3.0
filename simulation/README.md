# simulation/

| Folder | Contents |
|---|---|
| `engine/` | Order matching, fees, slippage, funding, position and P&L accounting. |
| `strategies/` | Decision rules that consume forecasts and emit orders. |
| `scenarios/` | Named market conditions to replay: crashes, low-volatility chop, regime shifts, liquidity droughts. |

**Costs are not optional.** Fees, slippage and funding belong in the engine before any strategy
number is quoted. A no-cost figure may be reported only next to its with-cost counterpart.

The engine must never see a bar the strategy has not been given first — that is the same
look-ahead problem as leakage, one layer up.
