"""
Baseline harness — the number everything else must beat.

Reads TRAIN and VALIDATION only. The test window is not opened.

Primary target : realized volatility over the next hour,
                 RV_t = sqrt(sum of squared 1-minute log-returns in hour t),
                 modelled in logs.
Secondary      : next-bar log-return, kept so the hard question stays measured.

Baselines
---------
    naive        RV_{t+1} = RV_t                  (the number to beat)
    rolling-24h  mean of the last 24 hourly RVs
    ewma         RiskMetrics-style, lambda = 0.94
    HAR          Corsi (2009), regressed on hour / day / week components

HAR is included because it is the standard benchmark in the realized-volatility
literature. A machine-learning model that does not beat a three-term linear
regression on this target has not earned its complexity.

Usage
-----
    python scripts/run_baselines.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.evaluation.metrics import qlike, summarize  # noqa: E402

TRAIN_END = "2024-01-01"
VAL_END = "2025-07-01"
MIN_BARS_PER_HOUR = 55        # an hour missing >5 minutes is not a clean RV
LAMBDA = 0.94


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def hourly_rv(df: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping hourly realized volatility from 1-minute log-returns.

    Gap-aware twice over: returns spanning a missing bar are dropped, and any
    hour left with fewer than MIN_BARS_PER_HOUR observations is discarded
    rather than reported as a low-volatility hour. An outage looks exactly like
    a calm hour if you do not check.
    """
    s = df.set_index("ts")["close"]
    r = np.log(s / s.shift(1))
    r = r[s.index.to_series().diff() == pd.Timedelta("1min")]

    g = r.groupby(pd.Grouper(freq="h"))
    rv = np.sqrt(g.apply(lambda x: float(np.sum(x ** 2))))
    n = g.size()
    out = pd.DataFrame({"rv": rv, "n": n}).dropna()
    return out[out["n"] >= MIN_BARS_PER_HOUR][["rv"]]


def har_design(log_rv: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Corsi HAR components, all strictly lagged so nothing sees its own hour."""
    d = pd.DataFrame({"y": log_rv})
    d["h"] = log_rv.shift(1)                       # previous hour
    d["d"] = log_rv.shift(1).rolling(24).mean()    # previous day
    d["w"] = log_rv.shift(1).rolling(168).mean()   # previous week
    d = d.dropna()
    X = np.column_stack([np.ones(len(d)), d["h"], d["d"], d["w"]])
    return X, d["y"].to_numpy(), d.index


def main() -> int:
    lines: list[str] = []

    def say(t=""):
        print(t)
        lines.append(t)

    src = data_root() / "interim" / "binance_BTCUSDT_1m.parquet"
    df = pd.read_parquet(src)

    bars_tr = df[df.ts < TRAIN_END]
    bars_va = df[(df.ts >= TRAIN_END) & (df.ts < VAL_END)]

    rv_tr, rv_va = hourly_rv(bars_tr), hourly_rv(bars_va)
    say(f"hourly RV   train {len(rv_tr):,}   validation {len(rv_va):,}")
    say(f"(hours dropped for <{MIN_BARS_PER_HOUR} clean minutes: "
          f"train {len(bars_tr) // 60 - len(rv_tr):,}, "
          f"validation {len(bars_va) // 60 - len(rv_va):,})\n")

    # Fit on train, evaluate on validation. HAR needs a week of history, so the
    # validation series is prefixed with the tail of train to warm the rolling
    # windows -- prefix rows are used as INPUTS only and never scored.
    warm = 168 + 1
    rv_all = pd.concat([rv_tr.iloc[-warm:], rv_va])
    log_tr = np.log(rv_tr["rv"])
    log_all = np.log(rv_all["rv"])

    Xtr, ytr, _ = har_design(log_tr)
    Xall, yall, idx_all = har_design(log_all)
    scored = idx_all >= rv_va.index[0]
    Xva, yva = Xall[scored], yall[scored]
    idx_va = idx_all[scored]

    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    say("HAR coefficients (log RV):")
    for name, b in zip(["const", "hour", "day", "week"], beta):
        say(f"  {name:6s} {b:+.4f}")
    say()

    # ------------------------------------------------------- predictions ---
    naive = Xva[:, 1]                               # previous hour, in logs
    har = Xva @ beta

    roll24 = log_all.shift(1).rolling(24).mean().reindex(idx_va).to_numpy()
    ewma = log_all.shift(1).ewm(alpha=1 - LAMBDA).mean().reindex(idx_va).to_numpy()

    preds = {"naive (RV_t)": naive, "rolling-24h": roll24,
             f"ewma (lam={LAMBDA})": ewma, "HAR": har}

    say("PRIMARY TARGET — log realized volatility, next hour")
    say(f"validation: {idx_va[0]:%Y-%m-%d} .. {idx_va[-1]:%Y-%m-%d}  "
          f"({len(yva):,} hours)\n")
    say(f"{'model':<20} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8} {'R2 vs naive':>12}")
    say("-" * 60)
    # RMSE/MAE/R2 are all reported in LOG space; QLIKE is a level-space loss by
    # definition, so it alone takes exp(). An earlier version printed log-space
    # RMSE beside a level-space R2, which made this table silently
    # incomparable with the LightGBM one.
    ss_base = float(np.sum((yva - naive) ** 2))
    for name, p in preds.items():
        ss = float(np.sum((yva - p) ** 2))
        r2 = ("  (baseline)" if name.startswith("naive")
              else f"{1 - ss / ss_base:>+12.4f}")
        say(f"{name:<20} {np.sqrt(np.mean((yva - p) ** 2)):>8.4f} "
            f"{np.mean(np.abs(yva - p)):>8.4f} "
            f"{qlike(np.exp(yva), np.exp(p)):>8.4f} {r2}")

    # -------------------------------------------- secondary: returns -------
    s = bars_va.set_index("ts")["close"]
    r = np.log(s / s.shift(1))
    r = r[s.index.to_series().diff() == pd.Timedelta("1min")].dropna()
    y_ret = r.to_numpy()[1:]
    zero = np.zeros_like(y_ret)
    drift = np.full_like(y_ret, float(np.log(bars_tr.close.iloc[-1] / bars_tr.close.iloc[0])
                                      / max(len(bars_tr) - 1, 1)))
    prev = r.to_numpy()[:-1]

    say("\n\nSECONDARY TARGET — next-bar log-return")
    say(f"validation: {len(y_ret):,} returns\n")
    say(f"{'model':<20} {'RMSE':>10} {'hit rate':>10} {'R2 vs zero':>12}")
    say("-" * 56)
    for name, p in {"zero": zero, "drift": drift, "previous return": prev}.items():
        m = summarize(y_ret, p, zero, volatility=False)
        r2 = "  (baseline)" if name == "zero" else f"{m['r2_vs_baseline']:>+12.6f}"
        hit = "        n/a" if name in ("zero", "drift") else f"{m['hit_rate']:>10.4f}"
        say(f"{name:<20} {m['rmse']:>10.6f} {hit} {r2}")

    say("\nRead this as: any model must beat 'naive' on the primary target and")
    say("'zero' on the secondary. Neither is a straw man.")

    out = ROOT / "forecasting" / "evaluation" / "BASELINES.md"
    out.write_text(
        "# Baseline results\n\n"
        "*Generated by `scripts/run_baselines.py`. Train and validation only; "
        "the test window is not read.*\n\n```\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
