"""
Strategy lab — momentum and volatility targeting, tested without leakage.

Rebuilds the one promising result from scratch after finding a lookahead bug in
the first attempt: the momentum signal had been computed from a cumulative sum
of FORWARD returns, so the running total at t already contained the return
being predicted. Here the signal is built from actual close prices.

The full grid is reported, never the best cell. Trying twenty configurations
and quoting the winner is precisely how false positives are manufactured
(Park & Irwin 2007), so every combination appears whether it flatters the
thesis or not.

Three questions
---------------
1. Does trend following beat buy-and-hold once the leak is removed?
2. How much of any edge is the MOMENTUM signal and how much is the VOLATILITY
   TARGETING? They are usually bundled and credited to the signal.
3. Does our LightGBM volatility forecast -- which genuinely beats HAR by 30% --
   size positions better than a plain ATR? That is the one route by which the
   forecasting work could pay for itself.

Usage
-----
    python scripts/strategy_lab.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.volatility_v2 import build as build_v2, hourly_base   # noqa: E402

LOOKBACKS = [30, 60, 90, 180, 365]        # days, fixed in advance
COSTS_BPS = [0, 5, 10, 20]
VOL_TARGET_ANN = 0.40                     # 40% annualised
MAX_LEVERAGE = 3.0
PPY = 365.0
FOLD_START = "2021-01-01"
BLOCK = pd.Timedelta(days=180)

LGB = dict(objective="regression", metric="rmse", learning_rate=0.03,
           num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
           verbosity=-1, seed=42)


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def perf(r: np.ndarray, turnover: np.ndarray) -> dict:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return dict(total=np.nan, sharpe=np.nan, maxdd=np.nan, turn=np.nan)
    cum = np.cumsum(r)
    return dict(
        total=float(np.exp(cum[-1]) - 1),
        sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(PPY)),
        maxdd=float((np.maximum.accumulate(cum) - cum).max()),
        turn=float(turnover.mean() * PPY),
    )


def run(pos: np.ndarray, fwd: np.ndarray, cost_bps: float) -> dict:
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    r = pos * fwd - turn * cost_bps / 1e4
    return perf(r, turn)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--skip-model", action="store_true",
                    help="skip the LightGBM vol-sizing rows (much faster)")
    args = ap.parse_args()

    lines: list[str] = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    bars = pd.read_parquet(data_root() / "interim" /
                       f"binance_{args.symbol}_1m.parquet")

    # ---- daily prices, computed from CLOSES, no forward information --------
    px = bars.set_index("ts")["close"].resample("D").last().dropna()
    logpx = np.log(px)
    fwd_ret = logpx.shift(-1) - logpx            # label: next day's log return

    # ---- realized daily vol from minute bars (backward looking) -----------
    h = hourly_base(bars)
    dvar = h["rv2"].resample("D").sum()
    dvol = np.sqrt(dvar).reindex(px.index)
    realized_vol_20 = dvol.rolling(20).mean()    # trailing, no leak

    # ---- our model's forecast of next-day volatility ----------------------
    if args.skip_model:
        fc = pd.Series(dtype=float)
        f24 = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    else:
        f24 = build_v2(bars, horizon=24)
    cols = [c for c in f24.columns if c != "y"]
    if not args.skip_model:
        fc = pd.Series(index=f24.index, dtype=float)
    start = pd.Timestamp(FOLD_START, tz="UTC")
    while (not args.skip_model) and start < f24.index.max():
        stop = start + BLOCK
        tr = f24[f24.index < start]
        te = f24[(f24.index >= start) & (f24.index < stop)]
        if len(tr) > 5000 and len(te) > 0:
            m = lgb.train(LGB, lgb.Dataset(tr[cols], tr["y"]), num_boost_round=300)
            fc.loc[te.index] = np.exp(m.predict(te[cols]))
        start = stop
    model_vol = (fc.resample("D").last().reindex(px.index)
             if not args.skip_model else pd.Series(np.nan, index=px.index))

    # ---- assemble ----------------------------------------------------------
    # The model-vol column only exists from 2021 (it needs a walk-forward
    # history first). Requiring it for everything would silently discard three
    # years of price data from the momentum tests, so it is dropped only where
    # it is actually used.
    d = pd.DataFrame({"logpx": logpx, "fwd": fwd_ret,
                      "atr_vol": realized_vol_20, "model_vol": model_vol})
    for L in LOOKBACKS:
        d[f"mom{L}"] = np.sign(logpx - logpx.shift(L))
    d_model = d.dropna()                       # 2021+, for model sizing only
    d = d.drop(columns="model_vol").dropna()   # full history otherwise
    fwd = d["fwd"].to_numpy()
    fwd_model = d_model["fwd"].to_numpy()

    say(f"# Strategy lab — {args.symbol}")
    say()
    say(f"**{len(d):,} daily bars**, {d.index.min():%Y-%m-%d} .. "
        f"{d.index.max():%Y-%m-%d}. Signals from closes only; the earlier "
        "lookahead bug is gone.")
    say()
    say(f"Vol targeting scales position to a {VOL_TARGET_ANN:.0%} annualised "
        f"target, capped at {MAX_LEVERAGE:.0f}x leverage.")
    say()

    def sizing(kind: str, frame: pd.DataFrame) -> np.ndarray:
        if kind == "constant":
            return np.ones(len(frame))
        v = frame["atr_vol" if kind == "ATR" else "model_vol"].to_numpy()
        v_ann = v * np.sqrt(PPY)
        s = np.divide(VOL_TARGET_ANN, v_ann, out=np.zeros_like(v_ann),
                      where=v_ann > 0)
        return np.clip(s, 0, MAX_LEVERAGE)

    # ------------------------------------------ 1. the decomposition -------
    say("## 1. Signal or sizing — which is doing the work?")
    say()
    say("Costs at 10 bps round trip. `always long` isolates the sizing effect;")
    say("`constant size` isolates the momentum effect. The `model` sizing rows")
    say(f"cover 2021 onward only ({len(d_model):,} bars), since the forecast "
        "needs a walk-forward history first; the others use all "
        f"{len(d):,} bars.")
    say()
    say("| position rule | sizing | bars | total | Sharpe | max DD | turnover/yr |")
    say("|---|---|---|---|---|---|---|")
    rows = {}
    for sig_name in ("always long", "momentum 365d"):
        for size_name in ("constant", "ATR", "model"):
            fr = d_model if size_name == "model" else d
            fw = fwd_model if size_name == "model" else fwd
            sig = (np.ones(len(fr)) if sig_name == "always long"
                   else fr["mom365"].clip(lower=0).to_numpy())
            p = sig * sizing(size_name, fr)
            r = run(p, fw, 10)
            rows[(sig_name, size_name)] = r
            say(f"| {sig_name} | {size_name} | {len(fr):,} | {r['total']:+.1%} | "
                f"{r['sharpe']:.2f} | {r['maxdd']:.2f} | {r['turn']:.1f}x |")
    say()

    bh = rows[("always long", "constant")]
    say(f"Buy-and-hold reference: {bh['total']:+.1%}, Sharpe {bh['sharpe']:.2f}, "
        f"max drawdown {bh['maxdd']:.2f}.")
    say()
    say("⚠ Compare `model` rows only against each other -- they run on a "
        "different, shorter window than the rest of the table.")
    say()

    # -------------------------------------------- 2. the full grid ---------
    say("## 2. Every lookback, every cost — the whole grid")
    say()
    say("Reported in full so nothing is cherry-picked. Sizing = ATR vol target.")
    say()
    say("| lookback | " + " | ".join(f"{c} bps" for c in COSTS_BPS) + " |")
    say("|---|" + "---|" * len(COSTS_BPS))
    for L in LOOKBACKS:
        p = d[f"mom{L}"].clip(lower=0).to_numpy() * sizing("ATR", d)
        cells = []
        for c in COSTS_BPS:
            r = run(p, fwd, c)
            cells.append(f"{r['sharpe']:.2f}")
        say(f"| {L}d | " + " | ".join(cells) + " |")
    say()
    say("Cells are Sharpe ratios. Buy-and-hold with the same vol targeting "
        f"scores {rows[('always long','ATR')]['sharpe']:.2f} at 10 bps.")
    say()

    # ------------------------------------------- 3. sub-periods ------------
    say("## 3. Does it survive being cut into regimes?")
    say()
    say("Every lookback, ATR vol targeting, 10 bps, against buy-and-hold sized")
    say("the same way. Sharpe ratios.")
    say()
    periods = (("2018-01-01", "2021-01-01", "2018-2020"),
               ("2021-01-01", "2023-01-01", "2021-2022"),
               ("2023-01-01", "2027-01-01", "2023-2026"))
    p_bh = np.ones(len(d)) * sizing("ATR", d)
    say("| lookback | " + " | ".join(p[2] for p in periods) + " | full |")
    say("|---|" + "---|" * (len(periods) + 1))
    for L in LOOKBACKS:
        p_str = d[f"mom{L}"].clip(lower=0).to_numpy() * sizing("ATR", d)
        cells = []
        for lo, hi, _ in periods:
            m = (d.index >= lo) & (d.index < hi)
            cells.append(f"{run(p_str[m], fwd[m], 10)['sharpe']:.2f}"
                         if m.sum() > 100 else "—")
        cells.append(f"**{run(p_str, fwd, 10)['sharpe']:.2f}**")
        say(f"| {L}d | " + " | ".join(cells) + " |")
    bh_cells = []
    for lo, hi, _ in periods:
        m = (d.index >= lo) & (d.index < hi)
        bh_cells.append(f"{run(p_bh[m], fwd[m], 10)['sharpe']:.2f}"
                        if m.sum() > 100 else "—")
    bh_cells.append(f"**{run(p_bh, fwd, 10)['sharpe']:.2f}**")
    say("| *buy & hold* | " + " | ".join(bh_cells) + " |")
    say()
    say("Drawdowns, same configurations:")
    say()
    say("| lookback | max drawdown |")
    say("|---|---|")
    for L in LOOKBACKS:
        p_str = d[f"mom{L}"].clip(lower=0).to_numpy() * sizing("ATR", d)
        say(f"| {L}d | {run(p_str, fwd, 10)['maxdd']:.2f} |")
    say(f"| *buy & hold* | {run(p_bh, fwd, 10)['maxdd']:.2f} |")
    say()

    se = np.sqrt(PPY / len(d))
    say(f"⚠ Sharpe standard error on the full sample is about {se:.2f}. "
        "Sub-period figures are far noisier still. Treat any single cell as "
        "indicative and the pattern across cells as the evidence.")
    say()

    out = (ROOT / "forecasting" / "evaluation" /
       f"STRATEGY-LAB-{args.symbol}.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
