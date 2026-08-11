"""Phase-2 charts: the horizon effect, and the uncertainty band at work."""

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
    ev = ROOT / "forecasting" / "evaluation"

    # ------------------------------------------------ 1. horizon effect ----
    horizons = ["1 hour", "4 hours", "24 hours"]
    r2 = [0.1865, 0.3011, 0.2795]

    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURFACE)
    style(ax)
    bars = ax.bar(horizons, r2, width=0.5, color=[SERIES[0], SERIES[1], SERIES[0]],
                  zorder=3)
    bars[1].set_color(SERIES[1])
    for x, v in zip(horizons, r2):
        ax.annotate(f"+{v:.1%}", (x, v), ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel("R² improvement over HAR", color=INK2, fontsize=10)
    ax.set_ylim(0, max(r2) * 1.22)
    ax.set_title("The machine-learning edge nearly doubles at four hours",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.annotate("walk-forward, 31,000+ scored hours per horizon",
                xy=(0, 1.005), xycoords="axes fraction", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "phase2-horizons.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --------------------------------------------- 2. the band at work -----
    arr = np.load(ev / "_v2_preds.npy")
    idx = pd.read_parquet(ev / "_v2_index.parquet")["ts"]
    y, mid, har, lo, hi = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])

    peak = int(np.argmax(y))
    a, b = max(peak - 120, 0), min(peak + 120, len(y))
    t = pd.to_datetime(idx.iloc[a:b])

    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(ax)
    ax.fill_between(t, np.exp(lo[a:b]) * 100, np.exp(hi[a:b]) * 100,
                    color=SERIES[0], alpha=0.16, zorder=2,
                    label="90% forecast band")
    ax.plot(t, np.exp(y[a:b]) * 100, color=INK2, linewidth=2.0,
            label="realized", zorder=4)
    ax.plot(t, np.exp(mid[a:b]) * 100, color=SERIES[0], linewidth=1.7,
            label="forecast (median)", zorder=5)
    ax.set_ylabel("hourly realized volatility (%)", color=INK2, fontsize=10)
    # Honest title: the band DOES widen ahead of the spike, and it still falls
    # far short of it. Claiming it "catches" the spike would be false -- the
    # band tops out near 1.5% against 6.6% realized.
    ax.set_title("The band widens ahead of the spike, and still misses it 4x over",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.annotate("realized 6.6%  ·  band top 1.5%", xy=(t.iloc[len(t) // 2 - 8], 5.4),
                fontsize=9, color=INK2)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "phase2-band.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------- 3. tail capture -----
    fig, ax = plt.subplots(figsize=(8.4, 4.2), facecolor=SURFACE)
    style(ax)
    labels = ["random\nchance", "our upper-quantile\nmodel"]
    vals = [1.0, 32.5]
    bs = ax.bar(labels, vals, width=0.45, color=[GRID, SERIES[1]], zorder=3)
    bs[0].set_edgecolor(INK2)
    for x, v in zip(labels, vals):
        ax.annotate(f"{v:.1f}%", (x, v), ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("share of the worst 1% of hours flagged", color=INK2, fontsize=10)
    ax.set_ylim(0, 40)
    ax.set_title("Extreme-hour detection: 32x better than chance, and still "
                 "missing two thirds",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out / "phase2-tail.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print("wrote phase2-horizons.png, phase2-band.png, phase2-tail.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
