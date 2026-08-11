"""Final charts: the arbitrage decay, and the two-asset replication."""

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


def data_root() -> Path:
    env = os.environ.get("FORESIGHT_DATA_ROOT")
    return Path(env).expanduser() if env else ROOT / "data"


def main() -> int:
    out = ROOT / "analysis" / "graphical"

    # ------------------------------------------ 1. the arbitrage decay -----
    f = pd.read_parquet(data_root() / "raw" / "binance_futures" /
                        "funding_BTCUSDT.parquet").set_index("ts")
    r = f["last_funding_rate"]
    yearly = r.groupby(r.index.year).mean() * 3 * 365 * 100

    # One colour: an earlier version shaded bars above 10%, which invented a
    # category the data does not have. The final year is partial (the archive
    # ends in July) and is labelled as such rather than compared like-for-like.
    last_year = int(yearly.index.max())
    labels = [f"{y}*" if y == last_year else str(y) for y in yearly.index]

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    style(ax)
    ax.bar(labels, yearly.values, width=0.6, color=SERIES[0], zorder=3)
    for x, v in zip(labels, yearly.values):
        ax.annotate(f"{v:.1f}%", (x, v), ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("annualised funding rate", color=INK2, fontsize=10)
    # Honest title: the compression is real but not monotonic -- 2024 bounced,
    # which coincides with the spot-ETF launch and the leverage demand it drew.
    ax.set_title("The easiest trade in crypto, compressing unevenly",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.annotate("Long spot, short perpetual, collect funding — no forecast "
                "needed.\n2021 paid 30.6%; the last twelve months paid 3.4%.\n"
                "2024's bounce coincides with the spot-ETF launch.",
                xy=(0.30, 0.66), xycoords="axes fraction",
                fontsize=9.5, color=INK2)
    ax.annotate("* partial year", xy=(0.985, -0.14), xycoords="axes fraction",
                fontsize=8, color=INK2, ha="right")
    ax.set_ylim(0, max(yearly.values) * 1.22)
    fig.tight_layout()
    fig.savefig(out / "final-funding-decay.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------- 2. the replication --------
    lookbacks = ["30d", "60d", "90d", "180d", "365d"]
    btc = [0.99, 0.92, 0.61, 0.59, 0.61]
    eth = [0.81, 0.73, 0.49, 0.31, 0.40]
    btc_bh, eth_bh = 0.45, 0.25

    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(lookbacks))
    w = 0.35
    ax.bar(x - w / 2 - 0.01, btc, width=w, color=SERIES[0], label="BTC", zorder=3)
    ax.bar(x + w / 2 + 0.01, eth, width=w, color=SERIES[1], label="ETH", zorder=3)
    # Draw the reference lines only across the plotted bars, so they stop
    # short of the labels instead of striking through them.
    x_end = len(lookbacks) - 0.45
    for lvl, col in ((btc_bh, SERIES[0]), (eth_bh, SERIES[1])):
        ax.plot([-0.55, x_end], [lvl, lvl], color=col, linewidth=1.4,
                linestyle=(0, (4, 3)), zorder=4)
    # Reserve clear space to the right of the last bar for the reference
    # labels; placing them over the plot collided with the bars and clipped.
    ax.set_xlim(-0.6, len(lookbacks) - 0.5 + 1.15)
    ax.annotate(f"BTC buy & hold  {btc_bh:.2f}",
                (len(lookbacks) - 0.35, btc_bh), xytext=(6, -3),
                textcoords="offset points", fontsize=9,
                color=SERIES[0], ha="left", va="center", fontweight="bold")
    ax.annotate(f"ETH buy & hold  {eth_bh:.2f}",
                (len(lookbacks) - 0.35, eth_bh), xytext=(6, -3),
                textcoords="offset points", fontsize=9,
                color=SERIES[1], ha="left", va="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lookbacks, fontsize=10, color=INK)
    ax.set_xlabel("momentum lookback", color=INK2, fontsize=10)
    ax.set_ylabel("Sharpe ratio, after 10 bps costs", color=INK2, fontsize=10)
    ax.set_title("Same rules, two assets, same shape",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out / "final-replication.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print("wrote final-funding-decay.png, final-replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
