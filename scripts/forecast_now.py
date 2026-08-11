"""
Live volatility forecast — the working end of the project.

Loads the saved production model and issues a forecast for the next window,
as a median plus a 90% band. Falls back to training on the spot if no saved
model exists.

Every forecast is appended to a log, so the system can later be scored on what
it actually predicted rather than on a backtest. A model nobody keeps score of
is a model nobody knows is still working.

    python scripts/train_production.py --horizon 4   # once
    python scripts/forecast_now.py --horizon 4       # any time after
    python scripts/forecast_now.py --update          # fetch new data first
    python scripts/forecast_now.py --score           # grade past forecasts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.volatility_v2 import build, hourly_base   # noqa: E402

QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
BASE = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, verbosity=-1, seed=42)
ROUNDS = 400


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def log_path() -> Path:
    return data_root() / "forecast_log.csv"


# ------------------------------------------------------------- scoring -----
def score_log(bars: pd.DataFrame) -> int:
    p = log_path()
    if not p.exists():
        print("no forecast log yet -- run a forecast first")
        return 1
    log = pd.read_csv(p, parse_dates=["issued_at", "window_start"])
    h = hourly_base(bars)
    rv2 = h["rv2"]

    rows = []
    for _, r in log.iterrows():
        w0 = pd.Timestamp(r["window_start"], tz="UTC")
        w1 = w0 + pd.Timedelta(hours=int(r["horizon_h"]) - 1)
        seg = rv2.loc[w0:w1]
        if len(seg) < int(r["horizon_h"]):
            continue                       # window not complete yet
        realized = float(np.sqrt(seg.sum())) * 100
        rows.append({**r.to_dict(), "realized_pct": realized,
                     "in_band": r["lo_pct"] <= realized <= r["hi_pct"],
                     "err": realized - r["mid_pct"]})

    if not rows:
        print(f"{len(log)} forecast(s) logged, none with a complete window yet")
        return 0

    d = pd.DataFrame(rows)
    print(f"scored {len(d)} of {len(log)} logged forecasts\n")
    print(f"  MAE                {np.mean(np.abs(d['err'])):.4f} pct points")
    print(f"  bias               {np.mean(d['err']):+.4f} (positive = "
          f"under-forecast)")
    print(f"  inside 90% band    {d['in_band'].mean():.1%}  (target 90%)")
    print()
    print(d[["window_start", "horizon_h", "mid_pct", "realized_pct",
             "in_band"]].tail(10).to_string(index=False))
    return 0


# ------------------------------------------------------------ forecast -----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4, choices=[1, 4, 24])
    ap.add_argument("--update", action="store_true",
                    help="fetch newly published months, then rebuild")
    ap.add_argument("--score", action="store_true",
                    help="grade previously logged forecasts against outcomes")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if args.update:
        for script in ("ingestion/exchanges/binance.py", "scripts/build_interim_bars.py"):
            subprocess.run([sys.executable, str(ROOT / script), "--apply"], check=True)

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")

    if args.score:
        return score_log(bars)

    mdir = data_root() / "models" / f"h{args.horizon}"
    saved = (mdir / "manifest.json").exists()

    live = build(bars, horizon=args.horizon, require_label=False)
    labelled = build(bars, horizon=args.horizon)
    origin = live.index[-1]
    assert origin > labelled.index[-1], "live row must be newer than last label"

    if saved:
        man = json.loads((mdir / "manifest.json").read_text())
        cols = man["features"]
        models = {k: lgb.Booster(model_file=str(mdir / f"{k}.txt")) for k in QUANTILES}
        print(f"loaded model  h{args.horizon}  trained through "
              f"{man['trained_on']['to'][:10]}  (commit {man['git_commit']})")
        hs = man["holdout_scores"]
        print(f"  holdout R² vs naive {hs['r2_vs_naive']:+.3f} | "
              f"90% band coverage {hs['coverage_90']:.1%}")
    else:
        print("no saved model -- training on the spot "
              "(run train_production.py to avoid this)")
        cols = [c for c in labelled.columns if c != "y"]
        models = {}
        for k, q in QUANTILES.items():
            models[k] = lgb.train(
                dict(BASE, objective="quantile", alpha=q, metric="quantile"),
                lgb.Dataset(labelled[cols], labelled["y"]), num_boost_round=ROUNDS)

    x = live[cols].iloc[[-1]]
    lo, mid, hi = (float(np.exp(models[k].predict(x)[0])) * 100
                   for k in ("q05", "q50", "q95"))
    last = float(np.exp(live["har_h"].iloc[-1])) * 100
    window_start = origin + pd.Timedelta(hours=1)

    print()
    print("=" * 64)
    print(f"  BTCUSDT realized volatility, next {args.horizon}h")
    print(f"  window {window_start:%Y-%m-%d %H:%M} .. "
          f"{window_start + pd.Timedelta(hours=args.horizon - 1):%H:%M} UTC")
    print("=" * 64)
    print(f"  last observed hour   {last:6.3f} %")
    print(f"  forecast (median)    {mid:6.3f} %")
    print(f"  90% band             {lo:6.3f} %  ..  {hi:6.3f} %")
    ratio = mid / last if last > 0 else float("nan")
    mood = ("calmer than" if ratio < 0.9 else
            "similar to" if ratio < 1.1 else "more volatile than")
    print(f"  -> {mood} the hour just passed ({ratio:.2f}x)")
    print("=" * 64)

    if not args.no_log:
        rec = pd.DataFrame([{
            "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window_start": window_start.isoformat(),
            "horizon_h": args.horizon,
            "last_pct": round(last, 4), "lo_pct": round(lo, 4),
            "mid_pct": round(mid, 4), "hi_pct": round(hi, 4),
            "model": "saved" if saved else "adhoc",
        }])
        p = log_path()
        rec.to_csv(p, mode="a", header=not p.exists(), index=False)
        print(f"\nlogged to {p}")
        print("Grade it later with:  python scripts/forecast_now.py --score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
