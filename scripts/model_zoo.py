"""
The full model shortlist, evaluated identically.

Everything from docs/MODEL-CONCEPTS.md, scored by the same code on the same
rows so the comparison means something:

    naive          carry the last hour forward
    HAR            Corsi (2009), the econometric benchmark
    Ridge          regularised linear on all features
    Random Forest  bagged trees
    Extra Trees    bagged trees with random split points
    XGBoost        boosted trees, the other major implementation
    LightGBM       boosted trees, leaf-wise
    MLP (small)    64-32, the shallow control
    MLP (deep)     256-128-64, the "does more depth help" control

Walk-forward: expanding window, refit every 90 days, each block scored before
it is ever trained on.

Usage
-----
    python scripts/model_zoo.py --horizon 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.volatility_v2 import build                              # noqa: E402
from forecasting.evaluation.diebold_mariano import diebold_mariano    # noqa: E402
from forecasting.evaluation.metrics import qlike                      # noqa: E402

FOLD_START = "2023-01-01"
BLOCK = pd.Timedelta(days=90)
HAR_COLS = ["har_h", "har_d", "har_w"]
SEED = 42


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def make_models():
    """Fresh estimators per fold; a refit must not inherit the last fold."""
    return {
        "Ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=0.5,
            n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=0.5,
            n_jobs=-1, random_state=SEED),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, tree_method="hist"),
        "LightGBM": "lgb",
        "MLP (64-32)": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
            early_stopping=True, n_iter_no_change=15, random_state=SEED)),
        "MLP (256-128-64)": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), alpha=1e-3, max_iter=300,
            early_stopping=True, n_iter_no_change=15, random_state=SEED)),
    }


LGB_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, seed=SEED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()

    lines: list[str] = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    f = build(bars, horizon=args.horizon)
    cols = [c for c in f.columns if c != "y"]

    names = list(make_models().keys()) + ["naive", "HAR"]
    preds: dict[str, list[np.ndarray]] = {n: [] for n in names}
    ys: list[np.ndarray] = []
    timing: dict[str, float] = {n: 0.0 for n in names}

    start = pd.Timestamp(FOLD_START, tz="UTC")
    folds = 0
    while start < f.index.max():
        stop = start + BLOCK
        tr = f[f.index < start]
        te = f[(f.index >= start) & (f.index < stop)]
        if len(tr) < 5000 or len(te) < 200:
            start = stop
            continue
        folds += 1
        ys.append(te["y"].to_numpy())

        preds["naive"].append(te["har_h"].to_numpy())
        X = np.column_stack([np.ones(len(tr)), tr[HAR_COLS].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)
        Xe = np.column_stack([np.ones(len(te)), te[HAR_COLS].to_numpy()])
        preds["HAR"].append(Xe @ beta)

        for name, est in make_models().items():
            t0 = time.time()
            if est == "lgb":
                m = lgb.train(LGB_PARAMS, lgb.Dataset(tr[cols], tr["y"]),
                              num_boost_round=400)
                preds[name].append(m.predict(te[cols]))
            else:
                est.fit(tr[cols], tr["y"])
                preds[name].append(est.predict(te[cols]))
            timing[name] += time.time() - t0
        print(f"  fold {folds}: {start:%Y-%m-%d} .. {stop:%Y-%m-%d} "
              f"({len(te):,} hours)", flush=True)
        start = stop

    y = np.concatenate(ys)
    P = {n: np.concatenate(v) for n, v in preds.items()}

    say(f"# Model zoo — {args.horizon}h realized volatility")
    say()
    say(f"Walk-forward, {folds} folds, **{len(y):,} scored hours**, "
        f"{len(cols)} features. Every model refit on each expanding window and "
        f"scored on the block that follows.")
    say()

    ss_n = float(np.sum((y - P["naive"]) ** 2))
    ss_h = float(np.sum((y - P["HAR"]) ** 2))
    order = sorted(P, key=lambda n: float(np.sum((y - P[n]) ** 2)))

    say("| model | RMSE | MAE | QLIKE | R² vs naive | R² vs HAR | fit time |")
    say("|---|---|---|---|---|---|---|")
    for n in order:
        p = P[n]
        ss = float(np.sum((y - p) ** 2))
        tm = f"{timing[n]:.0f}s" if timing.get(n) else "—"
        say(f"| {'**' + n + '**' if n == order[0] else n} | "
            f"{np.sqrt(np.mean((y - p) ** 2)):.4f} | "
            f"{np.mean(np.abs(y - p)):.4f} | "
            f"{qlike(np.exp(y), np.exp(p)):.4f} | "
            f"{1 - ss / ss_n:+.4f} | {1 - ss / ss_h:+.4f} | {tm} |")
    say()

    best = order[0]
    say(f"## Is **{best}** significantly better than the rest?")
    say()
    say("| challenger | DM stat | p-value | verdict |")
    say("|---|---|---|---|")
    for n in order[1:]:
        r = diebold_mariano(y, P[best], P[n])
        v = ("tie" if r["p_value"] >= 0.01
             else f"{best} better" if r["dm_stat"] < 0 else f"{n} better")
        say(f"| vs {n} | {r['dm_stat']:+.2f} | {r['p_value']:.1e} | {v} |")
    say()

    out = ROOT / "forecasting" / "evaluation" / f"MODEL-ZOO-{args.horizon}h.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
