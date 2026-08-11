"""
Phase-1 result charts: generalization, and what the forecast actually looks like.

Consumes the arrays written by scripts/evaluate_models.py so the charts can
never disagree with the numbers in PHASE1-RESULTS.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from features.volatility import build  # noqa: E402

# Validated categorical palette (dataviz reference instance, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def main() -> int:
    out = ROOT / "analysis" / "graphical"

    # --------------------------------------------- 1. generalization -------
    # Numbers copied from PHASE1-RESULTS.md, which evaluate_models.py generates.
    models = ["Ridge", "MLP", "LightGBM"]
    val = [0.0819, 0.0671, 0.1490]
    test = [0.0596, -0.0419, 0.1102]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(models))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, val, width=w, color=SERIES[0], label="validation", zorder=3)
    ax.bar(x + w / 2 + 0.01, test, width=w, color=SERIES[1], label="test", zorder=3)
    for xi, (v, t) in enumerate(zip(val, test)):
        ax.annotate(f"{v:+.3f}", (xi - w / 2 - 0.01, v), ha="center",
                    va="bottom" if v >= 0 else "top",
                    xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                    fontsize=9, color=INK2)
        ax.annotate(f"{t:+.3f}", (xi + w / 2 + 0.01, t), ha="center",
                    va="bottom" if t >= 0 else "top",
                    xytext=(0, 4 if t >= 0 else -4), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, color=INK)
    ax.set_ylabel("R² vs HAR  (0 = ties the benchmark)", color=INK2, fontsize=10)
    ax.set_title("Only LightGBM keeps its edge out of sample",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out / "phase1-generalization.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------- 2. forecast vs actual -----------
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    root = Path(env).expanduser() if env else ROOT / "data"
    f = build(pd.read_parquet(root / "interim" / "binance_BTCUSDT_1m.parquet"))
    te = f[f.index >= "2025-07-01"]
    arr = np.load(ROOT / "forecasting" / "evaluation" / "_test_preds.npy")
    y, naive, har, lgbm = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # a two-week slice containing the test period's most volatile hour
    peak = int(np.argmax(y))
    lo, hi = max(peak - 168, 0), min(peak + 168, len(y))
    idx = te.index[lo:hi]

    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(ax)
    ax.plot(idx, np.exp(y[lo:hi]) * 100, color=GRID, linewidth=2.4,
            label="realized", zorder=2)
    ax.plot(idx, np.exp(har[lo:hi]) * 100, color=SERIES[3], linewidth=1.6,
            label="HAR", zorder=3)
    ax.plot(idx, np.exp(lgbm[lo:hi]) * 100, color=SERIES[0], linewidth=1.8,
            label="LightGBM", zorder=4)
    ax.set_ylabel("hourly realized volatility (%)", color=INK2, fontsize=10)
    ax.set_title("Test period, two weeks around the most volatile hour",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "phase1-forecast-vs-actual.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print("wrote phase1-generalization.png, phase1-forecast-vs-actual.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
