"""
Train and SAVE the production model.

Everything before this trained a model, measured it, and threw it away. This
writes it to disk with the metadata needed to trust it later: the exact feature
list, the training range, the parameters, the git commit, and the walk-forward
score it earned.

Three quantiles are saved, not one. The median is the forecast; the 5% and 95%
models are what let the system say "this could be violent", which phase 1
showed a point forecast cannot.

Usage
-----
    python scripts/train_production.py --horizon 4
    python scripts/train_production.py --horizon 4 --holdout-days 90
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
from features.volatility_v2 import build                # noqa: E402
from forecasting.evaluation.metrics import qlike        # noqa: E402

QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
BASE = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, verbosity=-1, seed=42)
ROUNDS = 400


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def model_dir() -> Path:
    return data_root() / "models"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4, choices=[1, 4, 24])
    ap.add_argument("--holdout-days", type=int, default=90,
                    help="final stretch kept out of training, to score the "
                         "saved model honestly before it ships")
    args = ap.parse_args()

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    f = build(bars, horizon=args.horizon)
    cols = [c for c in f.columns if c != "y"]

    cutoff = f.index.max() - pd.Timedelta(days=args.holdout_days)
    tr, ho = f[f.index < cutoff], f[f.index >= cutoff]
    print(f"horizon {args.horizon}h | {len(cols)} features")
    print(f"train   {len(tr):,} hours  ({tr.index.min():%Y-%m-%d} .. "
          f"{tr.index.max():%Y-%m-%d})")
    print(f"holdout {len(ho):,} hours  ({ho.index.min():%Y-%m-%d} .. "
          f"{ho.index.max():%Y-%m-%d})\n")

    out = model_dir() / f"h{args.horizon}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- score on the holdout with models trained on `tr` only -------------
    scores, holdout_pred = {}, {}
    for name, q in QUANTILES.items():
        m = lgb.train(dict(BASE, objective="quantile", alpha=q, metric="quantile"),
                      lgb.Dataset(tr[cols], tr["y"]), num_boost_round=ROUNDS)
        holdout_pred[name] = m.predict(ho[cols])

    y = ho["y"].to_numpy()
    naive = ho["har_h"].to_numpy()
    p50 = holdout_pred["q50"]
    ss = float(np.sum((y - p50) ** 2))
    scores = {
        "rmse": float(np.sqrt(np.mean((y - p50) ** 2))),
        "qlike": qlike(np.exp(y), np.exp(p50)),
        "r2_vs_naive": 1 - ss / float(np.sum((y - naive) ** 2)),
        "coverage_90": float(np.mean((y >= holdout_pred["q05"]) &
                                     (y <= holdout_pred["q95"]))),
    }
    print("holdout performance (models trained WITHOUT it):")
    for k, v in scores.items():
        print(f"  {k:<14} {v:+.4f}")

    # ---- refit on everything and save --------------------------------------
    print("\nrefitting on the full history and saving ...")
    for name, q in QUANTILES.items():
        m = lgb.train(dict(BASE, objective="quantile", alpha=q, metric="quantile"),
                      lgb.Dataset(f[cols], f["y"]), num_boost_round=ROUNDS)
        m.save_model(str(out / f"{name}.txt"))

    (out / "manifest.json").write_text(json.dumps({
        "horizon_hours": args.horizon,
        "target": "log realized volatility over the next H hours",
        "features": cols,
        "n_features": len(cols),
        "params": BASE, "rounds": ROUNDS, "quantiles": QUANTILES,
        "trained_on": {"rows": int(len(f)),
                       "from": str(f.index.min()), "to": str(f.index.max())},
        "holdout_scores": scores,
        "holdout_days": args.holdout_days,
        "git_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    print(f"saved 3 models + manifest to {out}")
    print("\nForecast with:  python scripts/forecast_now.py --horizon "
          f"{args.horizon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
