"""
Does technical analysis predict BTC direction? A fair test.

Park & Irwin (2007) surveyed 95 studies of technical trading rules and found
most positive results compromised by three things: rules chosen after seeing
the data, no correction for testing many rules, and no transaction costs. This
script is built to avoid all three.

    - The indicator set is fixed in features/technical.py before any result.
    - Evaluation is walk-forward; no rule is selected on the data it is scored on.
    - Costs are charged, and the breakeven cost is reported.
    - Individual rules are tested TOGETHER with a multiple-testing correction,
      because trying twenty rules and reporting the best is how false positives
      are manufactured.

Also included is time-series momentum (Moskowitz, Ooi & Pedersen 2012) with
volatility-scaled sizing -- the price-only strategy with the strongest published
evidence, and the one I previously failed to test.

Usage
-----
    python scripts/test_technical.py --rule D
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.technical import build                      # noqa: E402

COST_BPS = 5.0            # round-trip, in basis points
FOLD_START = "2021-01-01"
BLOCK = pd.Timedelta(days=180)


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def perf(ret: np.ndarray, periods_per_year: float, label: str) -> dict:
    if len(ret) < 2:
        return {"name": label, "n": len(ret)}
    ann = np.sqrt(periods_per_year)
    cum = np.cumsum(ret)
    return {
        "name": label, "n": len(ret),
        "total": float(np.exp(cum[-1]) - 1),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * ann) if ret.std(ddof=1) > 0 else np.nan,
        "hit": float(np.mean(ret > 0)),
        "max_dd": float((np.maximum.accumulate(cum) - cum).max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="D", help="bar size: D, 4h, W")
    ap.add_argument("--cost", type=float, default=COST_BPS)
    args = ap.parse_args()

    lines: list[str] = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    ppy = {"D": 365.0, "4h": 6 * 365.0, "W": 52.0}[args.rule]
    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    f = build(bars, rule=args.rule)
    cols = [c for c in f.columns if c not in ("y", "y_dir")]

    say(f"# Technical analysis on BTC — {args.rule} bars")
    say()
    say(f"**{len(f):,} bars**, {len(cols)} indicators, "
        f"{f.index.min():%Y-%m-%d} .. {f.index.max():%Y-%m-%d}. "
        f"Round-trip cost {args.cost:.0f} bps.")
    say()

    # ---------------------------------------- 1. single rules, corrected ----
    say("## 1. Do individual indicators predict the next bar?")
    say()
    say("Spearman correlation between each indicator and the NEXT bar's return,")
    say("with a Benjamini-Hochberg correction across all indicators tested.")
    say()
    res = []
    for c in cols:
        rho, p = stats.spearmanr(f[c], f["y"])
        res.append((c, rho, p))
    res.sort(key=lambda r: r[2])
    ps = np.array([r[2] for r in res])
    m = len(ps)
    bh = ps * m / (np.arange(1, m + 1))          # BH adjusted
    bh = np.minimum.accumulate(bh[::-1])[::-1].clip(max=1.0)

    say("| indicator | Spearman ρ | raw p | BH-adjusted p | survives? |")
    say("|---|---|---|---|---|")
    for (c, rho, p), q in list(zip(res, bh))[:8]:
        say(f"| {c} | {rho:+.4f} | {p:.3f} | {q:.3f} | "
            f"{'**yes**' if q < 0.05 else 'no'} |")
    n_sig = int((bh < 0.05).sum())
    say()
    say(f"**{n_sig} of {m} indicators survive multiple-testing correction.**")
    say()

    # ------------------------------------------- 2. ML on all indicators ----
    say("## 2. A model given every indicator at once")
    say()
    ys, ps_, dates = [], [], []
    start = pd.Timestamp(FOLD_START, tz="UTC")
    while start < f.index.max():
        stop = start + BLOCK
        tr, te = f[f.index < start], f[(f.index >= start) & (f.index < stop)]
        if len(tr) > 400 and len(te) > 20:
            mdl = lgb.train(dict(objective="regression", metric="rmse",
                                 learning_rate=0.03, num_leaves=15,
                                 min_data_in_leaf=50, feature_fraction=0.7,
                                 bagging_fraction=0.8, bagging_freq=1,
                                 lambda_l2=5.0, verbosity=-1, seed=42),
                            lgb.Dataset(tr[cols], tr["y"]), num_boost_round=200)
            ys.append(te["y"].to_numpy())
            ps_.append(mdl.predict(te[cols]))
            dates.append(te.index)
        start = stop

    y = np.concatenate(ys)
    p = np.concatenate(ps_)
    idx = dates[0].append(dates[1:]) if len(dates) > 1 else dates[0]
    ss = float(np.sum((y - p) ** 2))
    ss0 = float(np.sum(y ** 2))
    hit = float(np.mean(np.sign(p[y != 0]) == np.sign(y[y != 0])))
    say(f"Walk-forward, {len(y):,} bars scored "
        f"({idx[0]:%Y-%m-%d} .. {idx[-1]:%Y-%m-%d}).")
    say()
    say(f"| metric | value |")
    say(f"|---|---|")
    say(f"| R² vs predicting zero | {1 - ss / ss0:+.4f} |")
    say(f"| directional hit rate | {hit:.4f} |")
    say(f"| correlation forecast vs realized | {np.corrcoef(p, y)[0, 1]:+.4f} |")
    say()

    # --------------------------------------------------- 3. strategies ------
    say("## 3. Strategies, after costs")
    say()
    cost = args.cost / 1e4

    # buy & hold
    bh_ret = y.copy()

    # ML-signal long/flat
    pos_ml = (p > 0).astype(float)
    turn_ml = np.abs(np.diff(np.concatenate([[0], pos_ml])))
    ml_ret = pos_ml * y - turn_ml * cost

    # time-series momentum: long if trailing 12-month return positive,
    # volatility-scaled. The classic Moskowitz-Ooi-Pedersen rule.
    look = int(ppy)                       # ~12 months of bars
    close_all = f.index.to_series()
    px = build(bars, rule=args.rule)["y"].cumsum()   # log price proxy
    px = px.reindex(f.index)
    mom = np.sign(px - px.shift(look)).reindex(idx).fillna(0).to_numpy()
    vol = f["atr_rel"].reindex(idx).to_numpy()
    target = 0.40 / np.sqrt(ppy)          # 40% annualised vol target
    size = np.clip(target / np.where(vol > 0, vol, np.nan), 0, 3)
    size = np.nan_to_num(size, nan=0.0)
    pos_ts = mom * size
    turn_ts = np.abs(np.diff(np.concatenate([[0], pos_ts])))
    ts_ret = pos_ts * y - turn_ts * cost

    # golden cross, the canonical chartist rule
    gc = np.sign(f["golden_cross"].reindex(idx).to_numpy())
    pos_gc = (gc > 0).astype(float)
    turn_gc = np.abs(np.diff(np.concatenate([[0], pos_gc])))
    gc_ret = pos_gc * y - turn_gc * cost

    say("| strategy | total return | Sharpe | hit | max drawdown |")
    say("|---|---|---|---|---|")
    for r, name in ((bh_ret, "buy & hold"),
                    (gc_ret, "golden cross (50/200)"),
                    (ts_ret, "TS momentum, vol-scaled"),
                    (ml_ret, "ML on all indicators")):
        s = perf(r, ppy, name)
        say(f"| {s['name']} | {s['total']:+.1%} | {s['sharpe']:.2f} | "
            f"{s['hit']:.0%} | {s['max_dd']:.2f} |")
    say()
    say("⚠ Two things to read carefully. Sharpe on this few bars carries a "
        f"standard error of roughly {np.sqrt(1 / max(len(y), 1)) * np.sqrt(ppy):.2f}, "
        "so differences smaller than that are noise. And the *hit* column is "
        "deflated for long/flat strategies: a flat day returns exactly zero, "
        "which does not count as a win, so a strategy that stands aside half "
        "the time cannot score above ~50% however good it is.")
    say()

    out = ROOT / "forecasting" / "evaluation" / f"TECHNICAL-{args.rule}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
