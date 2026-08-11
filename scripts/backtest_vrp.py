"""
Is the variance risk premium actually profitable after costs?

This is the only path from our data to a P&L, and it is worth being precise
about why. Our volatility forecast LOSES to DVOL (RMSE 0.311 vs 0.227), so
there is no edge in forecasting volatility better than the market. What survives
that finding is the premium itself: implied has averaged 4.20 vol points above
subsequent realized, in 67.5% of windows. Harvesting that requires no forecast.

So the strategy tested here is short variance, and the model is only asked
whether it can improve the TIMING -- sell when the premium looks unusually rich,
stand aside otherwise.

    A  always short      sell 30-day variance every month
    B  model-filtered    sell only when implied exceeds our forecast by a margin

What this backtest is NOT
-------------------------
Deribit lists no variance swaps. A real implementation is a delta-hedged
straddle, whose P&L is path-dependent, whose hedging costs scale with realized
volatility (exactly when you are losing), and which carries gamma risk this
model does not represent. Treat the numbers below as an UPPER BOUND on what the
premium offers, not as an achievable return.

Entries are monthly and non-overlapping, so the sample is ~64 trades over five
years, not 32,000 hours. Any Sharpe from 64 observations is a weak estimate.

Usage
-----
    python scripts/backtest_vrp.py --cost 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.volatility_v2 import build, hourly_base   # noqa: E402

HORIZON_H = 24 * 30
ANNUALIZE = 365 * 24
MIN_COVERAGE = 0.95
FOLD_START = "2023-01-01"
BLOCK = pd.Timedelta(days=90)

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.03,
              num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbosity=-1, seed=42)


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def load_dvol() -> pd.Series:
    raw = json.loads((data_root() / "raw" / "deribit" / "dvol_BTC_1h.json")
                     .read_text(encoding="utf-8"))
    d = pd.DataFrame(raw["data"], columns=raw["columns"])
    return pd.Series(d["close"].to_numpy(),
                     index=pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True),
                     name="dvol").sort_index()


def forward_rv(h: pd.DataFrame) -> pd.Series:
    var = h["rv2"].reindex(pd.date_range(h.index.min(), h.index.max(),
                                         freq="h", tz="UTC"))
    present = var.notna().astype(float)
    fwd = var.fillna(0).rolling(HORIZON_H).sum().shift(-HORIZON_H + 1)
    cover = present.rolling(HORIZON_H).mean().shift(-HORIZON_H + 1)
    return (np.sqrt(fwd / cover.where(cover >= MIN_COVERAGE)
                    * (ANNUALIZE / HORIZON_H)) * 100).dropna()


def stats(pnl: np.ndarray, label: str) -> dict:
    if len(pnl) == 0:
        return {"name": label, "n": 0}
    ann = np.sqrt(12)                       # monthly entries
    return {
        "name": label, "n": len(pnl),
        "total": float(pnl.sum()), "mean": float(pnl.mean()),
        "std": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "sharpe": float(pnl.mean() / pnl.std(ddof=1) * ann) if len(pnl) > 1 and pnl.std(ddof=1) > 0 else float("nan"),
        "win_rate": float(np.mean(pnl > 0)),
        "worst": float(pnl.min()),
        "max_dd": float((np.maximum.accumulate(pnl.cumsum()) - pnl.cumsum()).max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5,
                    help="round-trip cost in vol points")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="filter B: required implied-minus-forecast edge")
    args = ap.parse_args()

    lines: list[str] = []

    def say(t=""):
        print(t)
        lines.append(t)

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    h = hourly_base(bars)
    rv30 = forward_rv(h)
    dvol = load_dvol()

    # model forecast of 30-day realized vol, walk-forward
    f = build(bars, horizon=1, require_label=False)
    cols = [c for c in f.columns if c != "y"]
    df = f[cols].join(np.log(rv30).rename("y"), how="inner").dropna()
    df = df.join(dvol.rename("dvol"), how="inner")

    fc = pd.Series(index=df.index, dtype=float)
    start = pd.Timestamp(FOLD_START, tz="UTC")
    while start < df.index.max():
        stop = start + BLOCK
        tr = df[df.index < start]
        te = df[(df.index >= start) & (df.index < stop)]
        if len(tr) > 3000 and len(te) > 0:
            m = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr["y"]),
                          num_boost_round=300)
            fc.loc[te.index] = np.exp(m.predict(te[cols]))
        start = stop
    df["forecast"] = fc
    df = df.dropna(subset=["forecast"])

    # monthly, non-overlapping entries. The join above can leave a plain Index,
    # so rebuild a DatetimeIndex before grouping by period.
    df.index = pd.DatetimeIndex(df.index)
    month = df.index.tz_convert("UTC").to_period("M")
    entries = df.groupby(month).head(1)
    entries = entries.join(rv30.rename("realized"), how="inner")

    say("# Variance risk premium — is it profitable after costs?")
    say()
    say(f"Monthly non-overlapping entries: **{len(entries)} trades**, "
        f"{entries.index.min():%Y-%m} .. {entries.index.max():%Y-%m}.")
    say(f"Round-trip cost assumed: **{args.cost} vol points**.")
    say()
    say("P&L is quoted in *vol points* per trade: short variance earns")
    say("(implied − realized) − cost. Positive means the premium was collected.")
    say()

    gross = entries["dvol"].to_numpy() - entries["realized"].to_numpy()
    net_a = gross - args.cost

    edge = entries["dvol"].to_numpy() - entries["forecast"].to_numpy()
    take = edge >= args.margin
    net_b = net_a[take]

    say("| strategy | trades | total | mean/trade | win rate | worst | Sharpe | max DD |")
    say("|---|---|---|---|---|---|---|---|")
    for s in (stats(net_a, "A · always short"),
              stats(net_b, f"B · short only if edge ≥ {args.margin}")):
        if s["n"] == 0:
            say(f"| {s['name']} | 0 | — | — | — | — | — | — |")
            continue
        say(f"| {s['name']} | {s['n']} | {s['total']:+.1f} | {s['mean']:+.2f} | "
            f"{s['win_rate']:.0%} | {s['worst']:+.1f} | {s['sharpe']:.2f} | "
            f"{s['max_dd']:.1f} |")
    say()
    say(f"Gross premium before costs: {gross.mean():+.2f} vol points per trade "
        f"({np.mean(gross > 0):.0%} of trades positive).")
    say()

    # cost sensitivity -- the number that decides everything
    say("## How much cost kills it?")
    say()
    say("| round-trip cost | mean net | Sharpe | win rate |")
    say("|---|---|---|---|")
    for c in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        s = stats(gross - c, f"c{c}")
        say(f"| {c:.1f} vol pts | {s['mean']:+.2f} | {s['sharpe']:.2f} | "
            f"{s['win_rate']:.0%} |")
    say()

    say("## The tail, which is the whole risk")
    say()
    worst = np.sort(gross)[:5]
    say(f"Five worst trades, gross: {', '.join(f'{w:+.1f}' for w in worst)} vol points.")
    say(f"Mean of the worst 10%: {np.mean(np.sort(gross)[:max(1, len(gross)//10)]):+.1f}.")
    say()
    say("Short volatility loses most when volatility explodes, which is also")
    say("when a delta-hedged implementation bleeds hedging costs and when margin")
    say("is called. The proxy used here charges neither.")
    say()

    out = ROOT / "forecasting" / "evaluation" / "VRP-BACKTEST.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
