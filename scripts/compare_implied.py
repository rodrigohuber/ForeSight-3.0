"""
Does our forecast beat the MARKET's forecast?

Beating HAR is a research result. Beating Deribit's DVOL — the options market's
own 30-day implied volatility, priced by people doing exactly this — is a
trading result. This script settles the second question.

Three tests
-----------
1. VARIANCE RISK PREMIUM. Is implied systematically above subsequent realized?
   If so, selling volatility pays on average, and that is a tradeable edge that
   needs no forecasting skill at all.

2. FORECAST ACCURACY. Does DVOL or our model predict 30-day realized volatility
   more accurately?

3. ENCOMPASSING. Regress realized on DVOL alone, then on DVOL plus our
   forecast. If our forecast adds nothing once the market's price is known, we
   have no options edge -- whatever we scored against HAR.

Caveat carried throughout: 30-day forward windows sampled hourly OVERLAP almost
completely, so consecutive observations are anything but independent. Standard
errors are Newey-West with a 720-hour lag, and the effective sample is closer to
the ~64 non-overlapping months than to the 32,000 rows.

Usage
-----
    python scripts/compare_implied.py
"""

from __future__ import annotations

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

HORIZON_H = 24 * 30           # DVOL looks 30 days forward
MIN_COVERAGE = 0.95           # of expected hours inside the window
ANNUALIZE = 365 * 24          # hours per year
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
    ts = pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True)
    return pd.Series(d["close"].to_numpy(), index=ts, name="dvol").sort_index()


def forward_rv(h: pd.DataFrame) -> pd.Series:
    """Annualized realized volatility over the NEXT 30 days, in percent.

    Coverage-checked and coverage-scaled: a window missing hours would
    understate variance, so windows below MIN_COVERAGE are dropped and the rest
    are rescaled to a full window.
    """
    var = h["rv2"].reindex(
        pd.date_range(h.index.min(), h.index.max(), freq="h", tz="UTC"))
    present = var.notna().astype(float)
    fwd_var = var.fillna(0).rolling(HORIZON_H).sum().shift(-HORIZON_H + 1)
    cover = present.rolling(HORIZON_H).mean().shift(-HORIZON_H + 1)

    scaled = fwd_var / cover.where(cover >= MIN_COVERAGE)
    ann = np.sqrt(scaled * (ANNUALIZE / HORIZON_H)) * 100
    return ann.dropna()


def nw_tstat(x: np.ndarray, lag: int) -> float:
    """Newey-West t-statistic for a mean, honest about overlap."""
    n, m = len(x), float(np.mean(x))
    c = x - m
    lrv = float(c @ c) / n
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(c[k:] @ c[:-k]) / n
    return m / np.sqrt(max(lrv, 1e-300) / n)


def main() -> int:
    lines: list[str] = []

    def say(t=""):
        print(t)
        lines.append(t)

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    h = hourly_base(bars)
    rv30 = forward_rv(h)
    dvol = load_dvol()

    common = rv30.index.intersection(dvol.index)
    rv30, dv = rv30.loc[common], dvol.loc[common]

    say("# Implied vs realized — does the model beat the market?")
    say()
    say(f"Aligned hours: **{len(common):,}**  "
        f"({common.min():%Y-%m-%d} .. {common.max():%Y-%m-%d})")
    say()
    say("DVOL is Deribit's 30-day BTC implied volatility, annualized, in percent.")
    say("Realized is annualized volatility over the following 30 days, computed")
    say("from 1-minute bars, coverage-checked at 95%.")
    say()

    # ------------------------------------------- 1. variance risk premium ---
    vrp = (dv - rv30).to_numpy()
    t = nw_tstat(vrp, HORIZON_H)
    say("## 1. Variance risk premium")
    say()
    say("| quantity | value |")
    say("|---|---|")
    say(f"| mean implied (DVOL) | {dv.mean():.2f}% |")
    say(f"| mean subsequent realized | {rv30.mean():.2f}% |")
    say(f"| **mean premium (implied − realized)** | **{vrp.mean():+.2f} vol points** |")
    say(f"| share of windows where implied exceeded realized | "
        f"{np.mean(vrp > 0):.1%} |")
    say(f"| Newey-West t-stat (720h lag) | {t:.2f} |")
    say()
    if vrp.mean() > 0 and abs(t) > 2:
        say("**Implied volatility sits systematically above what follows.** That is")
        say("the variance risk premium, and harvesting it requires no forecasting")
        say("skill — only the willingness to be short volatility and survive the")
        say("windows where it inverts.")
    else:
        say("No statistically reliable premium in this sample.")
    say()

    # ---------------------------------------------- 2. forecast accuracy ----
    say("## 2. Who forecasts 30-day realized volatility better?")
    say()

    f = build(bars, horizon=1, require_label=False)
    cols = [c for c in f.columns if c != "y"]
    tgt = np.log(rv30)
    df = f[cols].join(tgt.rename("y"), how="inner").dropna()
    df = df.join(dv.rename("dvol"), how="inner")

    ys, ps, ds, ix = [], [], [], []
    start = pd.Timestamp(FOLD_START, tz="UTC")
    while start < df.index.max():
        stop = start + BLOCK
        tr = df[df.index < start]
        te = df[(df.index >= start) & (df.index < stop)]
        if len(tr) > 3000 and len(te) > 100:
            m = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr["y"]),
                          num_boost_round=300)
            ys.append(te["y"].to_numpy())
            ps.append(m.predict(te[cols]))
            ds.append(np.log(te["dvol"].to_numpy()))
            ix.append(te.index)
        start = stop

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    d = np.concatenate(ds)

    say(f"Walk-forward, {len(y):,} scored hours "
        f"({ix[0][0]:%Y-%m-%d} .. {ix[-1][-1]:%Y-%m-%d}). All in log space.")
    say()
    say("| forecaster | RMSE | bias |")
    say("|---|---|---|")
    say(f"| DVOL (the market) | {np.sqrt(np.mean((y - d) ** 2)):.4f} | "
        f"{np.mean(d - y):+.4f} |")
    say(f"| our model | {np.sqrt(np.mean((y - p) ** 2)):.4f} | "
        f"{np.mean(p - y):+.4f} |")
    say()

    # ------------------------------------------------- 3. encompassing ------
    say("## 3. Does our forecast add anything the market has not priced?")
    say()
    X1 = np.column_stack([np.ones(len(y)), d])
    X2 = np.column_stack([np.ones(len(y)), d, p])
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    r1 = 1 - np.sum((y - X1 @ b1) ** 2) / np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - X2 @ b2) ** 2) / np.sum((y - y.mean()) ** 2)

    resid = y - X2 @ b2
    se_scale = np.sqrt(np.sum(resid ** 2) / (len(y) - 3))
    say("Mincer-Zarnowitz encompassing regression of realized on the forecasts:")
    say()
    say("| specification | R² | coefficient on our forecast |")
    say("|---|---|---|")
    say(f"| realized ~ DVOL | {r1:.4f} | — |")
    say(f"| realized ~ DVOL + ours | {r2:.4f} | {b2[2]:+.4f} |")
    say()
    say(f"Incremental R² from our forecast: **{r2 - r1:+.4f}**")
    say()
    say("⚠ With 30-day windows sampled hourly, consecutive rows share 719 of 720")
    say("hours. The effective sample is nearer the ~64 non-overlapping months than")
    say("the row count, so treat any t-statistic here as indicative, not decisive.")
    say()

    out = ROOT / "forecasting" / "evaluation" / "IMPLIED-VS-MODEL.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(ROOT / "forecasting" / "evaluation" / "_implied.npy",
            np.column_stack([y, p, d]))
    pd.Series(ix[0].append(ix[1:]) if len(ix) > 1 else ix[0]).to_frame("ts") \
        .to_parquet(ROOT / "forecasting" / "evaluation" / "_implied_index.parquet",
                    index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
