"""
Does positioning predict direction? The last untested signal.

Binance publishes, every five minutes and for free, what the crowd is actually
holding: open interest, the long/short ratio of TOP traders by account and by
position size, the long/short ratio of all accounts, and the taker buy/sell
volume split. This is the one class of information a price-history model
structurally cannot see, and the reason it is worth testing last is that
everything derivable from price has already failed.

Two competing hypotheses, and they point opposite ways:

    SMART MONEY   top traders are informed; follow them.
    CROWDING      extreme positioning precedes squeezes; fade it.

Both are testable. Same protocol as everywhere else in this project: features
fixed before results are seen, Benjamini-Hochberg correction across all of
them, quintile buckets rather than a single correlation, and costs charged
before anything is called tradeable.

Usage
-----
    python scripts/test_positioning.py --bar 4h
"""

from __future__ import annotations

import argparse
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

COST_BPS = 10.0


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", default="4h", help="resample frequency")
    args = ap.parse_args()

    lines: list[str] = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    m = pd.read_parquet(data_root() / "raw" / "binance_futures" /
                        "metrics_BTCUSDT.parquet").set_index("ts").sort_index()
    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    px = bars.set_index("ts")["close"].resample(args.bar).last().dropna()

    g = m.resample(args.bar).last().dropna()
    f = pd.DataFrame(index=g.index)

    # --- levels -------------------------------------------------------------
    f["oi"] = np.log(g["sum_open_interest"])
    f["toptrader_acct"] = g["count_toptrader_long_short_ratio"]
    f["toptrader_pos"] = g["sum_toptrader_long_short_ratio"]
    f["all_accounts"] = g["count_long_short_ratio"]
    f["taker_ratio"] = g["sum_taker_long_short_vol_ratio"]

    # --- changes and standardised levels ------------------------------------
    win = int(pd.Timedelta("30D") / pd.Timedelta(args.bar))
    for c in ["oi", "toptrader_acct", "toptrader_pos", "all_accounts",
              "taker_ratio"]:
        f[f"{c}_chg"] = f[c].diff()
        f[f"{c}_z"] = (f[c] - f[c].rolling(win).mean()) / f[c].rolling(win).std()

    # --- crowding spread: top traders against everyone else -----------------
    f["smart_minus_crowd"] = f["toptrader_acct_z"] - f["all_accounts_z"]

    f["fwd"] = (np.log(px).diff().shift(-1)).reindex(f.index)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()

    per_year = pd.Timedelta("365D") / pd.Timedelta(args.bar)
    say(f"# Positioning as a directional signal — {args.bar} bars")
    say()
    say(f"**{len(f):,} bars**, {f.index.min():%Y-%m-%d} .. "
        f"{f.index.max():%Y-%m-%d}. Binance USD-M futures metrics, free, "
        f"published every five minutes.")
    say()

    cand = [c for c in f.columns if c != "fwd"]
    res = []
    for c in cand:
        rho, p = stats.spearmanr(f[c], f["fwd"])
        res.append((c, rho, p))
    res.sort(key=lambda r: r[2])
    ps = np.array([r[2] for r in res])
    k = len(ps)
    bh = np.minimum.accumulate((ps * k / np.arange(1, k + 1))[::-1])[::-1].clip(max=1)

    say("## Does any of it predict the next bar?")
    say()
    say(f"Spearman against the next bar's return, Benjamini-Hochberg corrected "
        f"across all {k} features.")
    say()
    say("| feature | ρ | raw p | BH p | survives? |")
    say("|---|---|---|---|---|")
    for (c, rho, p), q in list(zip(res, bh))[:10]:
        say(f"| {c} | {rho:+.4f} | {p:.3f} | {q:.3f} | "
            f"{'**yes**' if q < 0.05 else 'no'} |")
    n_sig = int((bh < 0.05).sum())
    say()
    say(f"**{n_sig} of {k} features survive correction.**")
    say()

    # ------------------------------------------------- quintile detail ------
    best = res[0][0]
    say(f"## Quintile detail on the strongest candidate (`{best}`)")
    say()
    d = f[[best, "fwd"]].copy()
    d["bucket"] = pd.qcut(d[best], 5, labels=["lowest", "2", "3", "4", "highest"])
    q = d.groupby("bucket", observed=True)["fwd"].agg(["mean", "count"])
    say("| bucket | mean next-bar return | n |")
    say("|---|---|---|")
    for b, row in q.iterrows():
        say(f"| {b} | {row['mean'] * 100:+.4f}% | {int(row['count']):,} |")
    spread = q["mean"].iloc[-1] - q["mean"].iloc[0]
    rt = 2 * COST_BPS / 100
    say()
    say(f"Top-minus-bottom spread: **{spread * 100:+.4f}%** per bar, against a "
        f"round-trip cost of {rt:.2f}%.")
    say(f"→ **{'clears' if abs(spread * 100) > rt else 'does NOT clear'} "
        f"its own transaction cost.**")
    say()

    # ------------------------------------------------------- strategy -------
    say("## As a strategy, after costs")
    say()
    say("| rule | total return | Sharpe | max DD |")
    say("|---|---|---|---|")
    fwd = f["fwd"].to_numpy()
    cost = COST_BPS / 1e4
    for name, pos in (
        ("buy & hold", np.ones(len(f))),
        (f"long when {best} high", (f[best] > f[best].median()).astype(float).to_numpy()),
        (f"long when {best} low", (f[best] < f[best].median()).astype(float).to_numpy()),
    ):
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        r = pos * fwd - turn * cost
        cum = np.cumsum(r)
        sh = (r.mean() / r.std(ddof=1) * np.sqrt(per_year)) if r.std(ddof=1) > 0 else np.nan
        dd = float((np.maximum.accumulate(cum) - cum).max())
        say(f"| {name} | {np.exp(cum[-1]) - 1:+.1%} | {sh:.2f} | {dd:.2f} |")
    say()
    say(f"⚠ Sharpe standard error here is about {np.sqrt(per_year / len(f)):.2f}, "
        "and this sample is only 2023 onward because Binance retains metrics "
        "files for a limited window.")
    say()

    out = ROOT / "forecasting" / "evaluation" / f"POSITIONING-{args.bar}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
