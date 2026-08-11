"""
Funding rate: as a directional signal, and as an income stream.

Two quite different uses of the same data, and the second is the interesting one.

1. AS A SIGNAL. Positive funding means longs pay shorts, i.e. the crowd is
   positioned long. The standard hypothesis is that extremes are contrarian:
   crowded longs precede squeezes. Tested here against next-period returns with
   the same rigour applied to the technical indicators.

2. AS INCOME — the cash-and-carry trade. Long spot, short the perpetual, net
   delta about zero, and collect funding whenever it is positive. This needs no
   forecast of anything. On this sample funding has been positive 85.7% of
   intervals, averaging roughly 11.9% annualised, which is a large number to
   leave on the table if it survives costs.

What the carry test does and does not model
-------------------------------------------
It charges entry and exit costs on both legs and assumes the position is held
continuously. It does NOT model: margin requirements and the capital they tie
up, liquidation risk if the basis gaps, exchange counterparty risk, or the
borrow cost of the spot leg if bought on margin. Treat the result as the gross
carry available, from which real-world frictions still have to come.

Usage
-----
    python scripts/test_funding.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0            # round trip, per leg
PPY = 365.0


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def main() -> int:
    lines: list[str] = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    fund = pd.read_parquet(data_root() / "raw" / "binance_futures" /
                           "funding_BTCUSDT.parquet").set_index("ts")
    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    px = bars.set_index("ts")["close"]

    say("# Funding rate — signal, and income")
    say()
    say(f"{len(fund):,} funding settlements, {fund.index.min():%Y-%m-%d} .. "
        f"{fund.index.max():%Y-%m-%d}, every 8 hours.")
    say()

    r = fund["last_funding_rate"]
    ann = r.mean() * 3 * PPY
    say("| statistic | value |")
    say("|---|---|")
    say(f"| mean funding per 8h | {r.mean() * 100:+.4f}% |")
    say(f"| annualised | **{ann * 100:+.1f}%** |")
    say(f"| intervals positive | {(r > 0).mean():.1%} |")
    say(f"| median | {r.median() * 100:+.4f}% |")
    say(f"| worst single settlement | {r.min() * 100:+.3f}% |")
    say()
    say("Positive funding means longs pay shorts. It has been positive in the")
    say("large majority of intervals, which is a structural cost of holding")
    say("leveraged long exposure and a structural income for the other side.")
    say()

    # ------------------------------------------- 1. directional signal -----
    say("## 1. Does funding predict the next move?")
    say()
    fwd8 = (np.log(px.resample("8h").last()).diff().shift(-1)
            .reindex(r.index))
    z = (r - r.rolling(90).mean()) / r.rolling(90).std()
    d = pd.DataFrame({"funding": r, "z": z, "fwd": fwd8}).dropna()

    say("| predictor | Spearman ρ | p-value | reading |")
    say("|---|---|---|---|")
    for name, col in (("funding level", "funding"), ("funding z-score (90p)", "z")):
        rho, p = stats.spearmanr(d[col], d["fwd"])
        verdict = ("contrarian, as theory predicts" if rho < 0 and p < 0.05
                   else "momentum-aligned" if rho > 0 and p < 0.05
                   else "no reliable relationship")
        say(f"| {name} | {rho:+.4f} | {p:.3f} | {verdict} |")
    say()

    # quintile analysis -- more informative than a single correlation
    d["bucket"] = pd.qcut(d["z"], 5, labels=["most negative", "2", "3", "4",
                                             "most positive"])
    g = d.groupby("bucket", observed=True)["fwd"].agg(["mean", "count"])
    say("Next-8h return by funding z-score bucket:")
    say()
    say("| funding bucket | mean next 8h return | n |")
    say("|---|---|---|")
    for b, row in g.iterrows():
        say(f"| {b} | {row['mean'] * 100:+.4f}% | {int(row['count']):,} |")
    say()
    spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
    say(f"Top-minus-bottom bucket spread: **{spread * 100:+.4f}%** per 8 hours. "
        f"A round trip costs {2 * COST_BPS / 100:.2f}%, so the spread would "
        f"need to exceed that to be tradeable as a signal.")
    say()

    # --------------------------------------------- 2. the carry trade ------
    say("## 2. The cash-and-carry trade — income without a forecast")
    say()
    say("Long spot, short the perpetual, net delta about zero. Collect funding")
    say("when positive, pay it when negative. No view on direction required.")
    say()

    # funding accrues to the SHORT perp leg when positive
    carry = r.copy()
    n_settle = len(carry)
    years = (carry.index[-1] - carry.index[0]).days / 365.25

    say("| entry+exit cost (both legs) | gross annualised | net annualised |")
    say("|---|---|---|")
    for c_bps in (0, 10, 20, 50):
        total_cost = 2 * c_bps / 1e4                      # in and out, 2 legs
        gross = carry.sum()
        net = gross - total_cost
        say(f"| {c_bps} bps | {gross / years * 100:+.1f}% | "
            f"{net / years * 100:+.1f}% |")
    say()
    say("Costs are one-off on a continuously held position, so they barely dent")
    say("a multi-year hold. The risk is not cost, it is that funding inverts.")
    say()

    # drawdown of the carry stream
    cum = carry.cumsum()
    dd = (np.maximum.accumulate(cum) - cum)
    worst_year = carry.resample("YE").sum().min()
    say("| carry-stream risk | value |")
    say("|---|---|")
    say(f"| worst drawdown in cumulative funding | {dd.max() * 100:.2f}% |")
    say(f"| worst calendar year | {worst_year * 100:+.1f}% |")
    say(f"| negative-funding stretches over 3 days | "
        f"{int(((r < 0).astype(int).groupby((r >= 0).cumsum()).sum() >= 9).sum())} |")
    say(f"| annualised volatility of the carry stream | "
        f"{carry.std() * np.sqrt(3 * PPY) * 100:.1f}% |")
    say(f"| **carry Sharpe (funding only)** | "
        f"**{carry.mean() / carry.std() * np.sqrt(3 * PPY):.2f}** |")
    say()
    say("⚠ A Sharpe near 17 should be disbelieved on sight. It is computed on")
    say("the funding stream ALONE, assuming a perfect and free delta hedge.")
    say("Real basis trades carry tracking error, and the Sharpe falls fast:")
    say()
    mu = r.mean() * 3 * PPY
    sd0 = r.std() * np.sqrt(3 * PPY)
    say("| annualised hedge tracking error | resulting Sharpe |")
    say("|---|---|")
    for te in (0.0, 0.02, 0.05, 0.10, 0.20):
        say(f"| {te * 100:.0f}% | {mu / np.sqrt(sd0 ** 2 + te ** 2):.2f} |")
    say()

    # ------------------------------------- 3. is the carry still there? ----
    say("## 3. ⚠ The carry is being arbitraged away")
    say()
    say("The multi-year average is dominated by 2020-21. What matters is what")
    say("is available now.")
    say()
    say("| year | annualised funding | intervals positive |")
    say("|---|---|---|")
    for y, g in r.groupby(r.index.year):
        say(f"| {y} | {g.mean() * 3 * PPY * 100:+.1f}% | {(g > 0).mean():.0%} |")
    last12 = r[r.index >= r.index.max() - pd.Timedelta(days=365)]
    say()
    say(f"**Trailing twelve months: {last12.mean() * 3 * PPY * 100:+.1f}% "
        f"annualised**, positive {(last12 > 0).mean():.0%} of intervals.")
    say()
    say("From 30.6% in 2021 to roughly 2-3% now. The trade was real and it has")
    say("largely been competed away. What remains does not obviously compensate")
    say("for exchange counterparty risk, margin capital and liquidation risk —")
    say("a government bond pays comparably with none of those.")
    say()
    say("This is worth stating plainly because it is the pattern of the whole")
    say("project: every edge found in this data was real at some point and has")
    say("since been arbitraged. That is what an efficient market looks like")
    say("from the inside.")
    say()

    out = ROOT / "forecasting" / "evaluation" / "FUNDING.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
