"""
Phase 2 — jump-robust features, multi-horizon, and quantile forecasts.

Answers the three things phase 1 left open:

  1. Does the jump/continuous decomposition plus HARQ measurement-error terms
     beat the phase-1 feature set?
  2. Does the edge hold at 4h and 24h, which matter more operationally than 1h?
  3. Can the model say anything useful about the TAIL, which phase 1 missed
     badly (6.6% realized against a 1.2% point forecast)?

(3) is answered by fitting quantile objectives rather than only the mean, and
scoring them with pinball loss and empirical coverage. A point forecast that
cannot flag "this hour could be violent" is not useful for risk; an upper
quantile can be.

Evaluation is WALK-FORWARD: the model is refit on an expanding window and
scored on the next block, repeatedly. That is a stricter test than one split,
and it also means the reported number does not depend on a single lucky
period.

Usage
-----
    python scripts/train_v2.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.volatility_v2 import build as build_v2                 # noqa: E402
from features import volatility as v1                                # noqa: E402
from forecasting.evaluation.diebold_mariano import diebold_mariano   # noqa: E402
from forecasting.evaluation.metrics import qlike                     # noqa: E402

HORIZONS = [1, 4, 24]
QUANTILES = [0.05, 0.50, 0.95]
HAR_COLS = ["har_h", "har_d", "har_w"]

BASE = dict(objective="regression", metric="rmse", learning_rate=0.03,
            num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            verbosity=-1, seed=42)

# Walk-forward: first fit uses everything before FOLD_START, then each fold
# adds the block just scored. Blocks are 90 days.
FOLD_START = "2023-01-01"
BLOCK = pd.Timedelta(days=90)


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def har_fit_predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(tr)), tr[HAR_COLS].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)
    Xe = np.column_stack([np.ones(len(te)), te[HAR_COLS].to_numpy()])
    return Xe @ beta


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def walk_forward(f: pd.DataFrame, cols: list[str], *, quantile: float | None = None):
    """Expanding-window walk-forward. Returns (y, pred, har, index)."""
    ys, ps, hs, ix = [], [], [], []
    start = pd.Timestamp(FOLD_START, tz="UTC")
    end = f.index.max()
    while start < end:
        stop = start + BLOCK
        tr = f[f.index < start]
        te = f[(f.index >= start) & (f.index < stop)]
        if len(te) < 200 or len(tr) < 5000:
            start = stop
            continue
        params = dict(BASE)
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile, metric="quantile")
        model = lgb.train(params, lgb.Dataset(tr[cols], tr["y"]),
                          num_boost_round=400)
        ys.append(te["y"].to_numpy())
        ps.append(model.predict(te[cols]))
        hs.append(har_fit_predict(tr, te))
        ix.append(te.index)
        start = stop
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(hs), ix[0].append(ix[1:]) if len(ix) > 1 else ix[0])


def main() -> int:
    lines: list[str] = []

    def say(t=""):
        print(t)
        lines.append(t)

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")

    say("# Phase 2 — jump-robust features, multi-horizon, quantiles")
    say()
    say("Walk-forward evaluation: expanding window, refit every 90 days, each")
    say("block scored before it is ever trained on. Every number below is")
    say("out-of-sample by construction.")
    say()

    results = {}

    # ============================================ 1. v1 vs v2 at h=1 ========
    say("## 1. Does feature set v2 beat v1?")
    say()
    f1 = v1.build(bars)
    c1 = [c for c in f1.columns if c != "y"]
    y1, p1, h1, _ = walk_forward(f1, c1)

    f2 = build_v2(bars, horizon=1)
    c2 = [c for c in f2.columns if c != "y"]
    y2, p2, h2, idx2 = walk_forward(f2, c2)

    say(f"v1: {len(c1)} features, {len(y1):,} scored hours")
    say(f"v2: {len(c2)} features, {len(y2):,} scored hours")
    say()
    say("| feature set | RMSE | QLIKE | R2 vs HAR |")
    say("|---|---|---|---|")
    for name, y, p, h in (("v1 (phase 1)", y1, p1, h1), ("v2 (jump + HARQ)", y2, p2, h2)):
        ss = float(np.sum((y - p) ** 2))
        ssh = float(np.sum((y - h) ** 2))
        say(f"| {name} | {np.sqrt(np.mean((y - p) ** 2)):.4f} | "
            f"{qlike(np.exp(y), np.exp(p)):.4f} | {1 - ss / ssh:+.4f} |")
    say()

    # ================================================ 2. horizons ===========
    say("## 2. Does the edge hold at longer horizons?")
    say()
    say("| horizon | hours scored | HAR RMSE | LightGBM RMSE | R2 vs HAR | DM p |")
    say("|---|---|---|---|---|---|")
    for H in HORIZONS:
        fh = build_v2(bars, horizon=H)
        ch = [c for c in fh.columns if c != "y"]
        y, p, h, _ = walk_forward(fh, ch)
        ss, ssh = float(np.sum((y - p) ** 2)), float(np.sum((y - h) ** 2))
        dm = diebold_mariano(y, p, h)
        results[f"h{H}"] = {"r2_har": 1 - ss / ssh, "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
        say(f"| {H}h | {len(y):,} | {np.sqrt(np.mean((y - h) ** 2)):.4f} | "
            f"{np.sqrt(np.mean((y - p) ** 2)):.4f} | {1 - ss / ssh:+.4f} | "
            f"{dm['p_value']:.1e} |")
    say()

    # ================================================ 3. quantiles ==========
    say("## 3. The tail — quantile forecasts")
    say()
    say("Phase 1's point forecast missed the largest spike by 5x. A point")
    say("forecast cannot express 'this hour could be violent'; an upper")
    say("quantile can. Scored with pinball loss and empirical coverage.")
    say()
    qpreds = {}
    for q in QUANTILES:
        yq, pq, _, _ = walk_forward(f2, c2, quantile=q)
        qpreds[q] = (yq, pq)
    say("| quantile | pinball loss | empirical coverage | target |")
    say("|---|---|---|---|")
    for q in QUANTILES:
        yq, pq = qpreds[q]
        cov = float(np.mean(yq <= pq))
        say(f"| {q:.2f} | {pinball(yq, pq, q):.5f} | {cov:.3f} | {q:.2f} |")
    say()

    y95, p95 = qpreds[0.95]
    y05, p05 = qpreds[0.05]
    inside = float(np.mean((y95 >= p05) & (y95 <= p95)))
    say(f"**90% interval covers {inside:.1%} of realized hours** (target 90%).")
    say()

    # how well does the upper quantile flag the worst hours?
    thresh = np.quantile(y95, 0.99)
    extreme = y95 >= thresh
    flagged = p95 >= np.quantile(p95, 0.99)
    recall = float(np.mean(flagged[extreme]))
    say(f"Of the worst 1% of hours, the 0.95-quantile model's own top-1% "
        f"forecasts flag **{recall:.1%}**.")
    say()

    out = ROOT / "forecasting" / "evaluation" / "PHASE2-RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    np.save(ROOT / "forecasting" / "evaluation" / "_v2_preds.npy",
            np.column_stack([y2, p2, h2, qpreds[0.05][1], qpreds[0.95][1]]))
    pd.Series(idx2).to_frame("ts").to_parquet(
        ROOT / "forecasting" / "evaluation" / "_v2_index.parquet", index=False)
    (ROOT / "forecasting" / "evaluation" / "_v2_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
