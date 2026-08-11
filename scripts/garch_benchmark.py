"""
GARCH family vs HAR vs LightGBM — the consecrated benchmarks, head to head.

The volatility literature's canonical models are GARCH (Bollerslev 1986),
GJR-GARCH (Glosten-Jagannathan-Runkle 1993, which adds the leverage asymmetry),
EGARCH (Nelson 1991, which models log-variance so it cannot go negative), and
HAR (Corsi 2009). This script fits all of them on the same rows as the machine
learning model so the comparison is like for like.

GARCH is fitted on hourly RETURNS and forecasts next-hour conditional variance.
HAR and LightGBM are fitted on realized volatility directly. That difference is
inherent to the model classes, not a handicap imposed here: GARCH infers latent
volatility from returns, while realized-volatility models observe it.

Usage
-----
    python scripts/garch_benchmark.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.volatility_v2 import build, hourly_base                 # noqa: E402
from forecasting.evaluation.diebold_mariano import diebold_mariano    # noqa: E402
from forecasting.evaluation.metrics import qlike                      # noqa: E402

FOLD_START = "2023-01-01"
BLOCK = pd.Timedelta(days=90)
HAR_COLS = ["har_h", "har_d", "har_w"]
GARCH_WINDOW = 8000          # hours of returns per refit; GARCH is slow

LGB_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, seed=42)

SPECS = {
    "GARCH(1,1)":  dict(vol="GARCH", p=1, q=1, o=0),
    "GJR-GARCH":   dict(vol="GARCH", p=1, q=1, o=1),
    "EGARCH(1,1)": dict(vol="EGARCH", p=1, q=1, o=1),
}


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def main() -> int:
    lines: list[str] = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    bars = pd.read_parquet(data_root() / "interim" / "binance_BTCUSDT_1m.parquet")
    h = hourly_base(bars)
    f = build(bars, horizon=1)
    cols = [c for c in f.columns if c != "y"]

    # hourly returns, in percent, for the GARCH models
    ret = (h["ret"] * 100).reindex(f.index).dropna()
    f = f.loc[ret.index]

    names = ["naive", "HAR", "LightGBM"] + list(SPECS)
    P: dict[str, list[np.ndarray]] = {n: [] for n in names}
    ys: list[np.ndarray] = []

    start = pd.Timestamp(FOLD_START, tz="UTC")
    fold = 0
    while start < f.index.max():
        stop = start + BLOCK
        tr = f[f.index < start]
        te = f[(f.index >= start) & (f.index < stop)]
        if len(tr) < 5000 or len(te) < 200:
            start = stop
            continue
        fold += 1
        ys.append(te["y"].to_numpy())
        P["naive"].append(te["har_h"].to_numpy())

        X = np.column_stack([np.ones(len(tr)), tr[HAR_COLS].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)
        P["HAR"].append(np.column_stack(
            [np.ones(len(te)), te[HAR_COLS].to_numpy()]) @ beta)

        m = lgb.train(LGB_PARAMS, lgb.Dataset(tr[cols], tr["y"]),
                      num_boost_round=400)
        P["LightGBM"].append(m.predict(te[cols]))

        # GARCH: fit once per fold on the tail of the training returns, then
        # roll forward one step at a time WITHOUT refitting (standard practice;
        # refitting hourly for 3 specs x 14 folds is not tractable)
        r_tr = ret.loc[:tr.index[-1]].iloc[-GARCH_WINDOW:]
        r_full = ret.loc[:te.index[-1]]
        for spec_name, kw in SPECS.items():
            try:
                res = arch_model(r_tr, mean="Constant", dist="t", **kw).fit(disp="off")
                # Re-filter the FULL series with the fitted parameters held
                # fixed. That yields a one-step conditional volatility at every
                # point without refitting, which keeps three specs x fourteen
                # folds tractable.
                fixed = arch_model(r_full, mean="Constant", dist="t", **kw)
                sim = fixed.fix(res.params)
                cv = pd.Series(np.asarray(sim.conditional_volatility),
                               index=r_full.index)
                pred = np.log(cv.reindex(te.index).clip(lower=1e-6) / 100)
                P[spec_name].append(pred.to_numpy())
            except Exception as e:
                print(f"    {spec_name} failed on fold {fold}: {e}")
                P[spec_name].append(np.full(len(te), np.nan))
        print(f"  fold {fold}: {start:%Y-%m-%d} ({len(te):,} hours)", flush=True)
        start = stop

    y = np.concatenate(ys)
    P = {n: np.concatenate(v) for n, v in P.items()}
    ok = np.all([np.isfinite(v) for v in P.values()], axis=0)
    y = y[ok]
    P = {n: v[ok] for n, v in P.items()}

    say("# GARCH family vs HAR vs LightGBM")
    say()
    say(f"Walk-forward, {fold} folds, **{len(y):,} scored hours**, "
        f"next-hour log realized volatility.")
    say()
    say("GARCH parameters are estimated once per fold on the preceding 8,000")
    say("hours, then held fixed while the conditional variance is filtered")
    say("forward. That is standard practice and avoids refitting hourly.")
    say()

    ss_n = float(np.sum((y - P["naive"]) ** 2))
    ss_h = float(np.sum((y - P["HAR"]) ** 2))
    order = sorted(P, key=lambda n: float(np.sum((y - P[n]) ** 2)))

    say("| model | RMSE | MAE | QLIKE | R² vs naive | R² vs HAR |")
    say("|---|---|---|---|---|---|")
    for n in order:
        p = P[n]
        ss = float(np.sum((y - p) ** 2))
        label = f"**{n}**" if n == order[0] else n
        say(f"| {label} | {np.sqrt(np.mean((y - p) ** 2)):.4f} | "
            f"{np.mean(np.abs(y - p)):.4f} | {qlike(np.exp(y), np.exp(p)):.4f} | "
            f"{1 - ss / ss_n:+.4f} | {1 - ss / ss_h:+.4f} |")
    say()

    best = order[0]
    say(f"## Diebold-Mariano, {best} against each")
    say()
    say("| challenger | DM | p | verdict |")
    say("|---|---|---|---|")
    for n in order[1:]:
        r = diebold_mariano(y, P[best], P[n])
        v = ("tie" if r["p_value"] >= 0.01
             else f"{best} better" if r["dm_stat"] < 0 else f"{n} better")
        say(f"| vs {n} | {r['dm_stat']:+.2f} | {r['p_value']:.1e} | {v} |")
    say()

    out = ROOT / "forecasting" / "evaluation" / "GARCH-BENCHMARK.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
