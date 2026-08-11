"""
Hourly feature matrix for forecasting next-hour realized volatility.

Every feature is built from 1-minute bars and is STRICTLY LAGGED: the row
stamped at hour t contains only quantities computable by the end of hour t, and
the label is RV for hour t+1. Nothing reaches across that boundary.

Feature families, and why each is here
--------------------------------------
HAR components      the three terms Corsi's regression uses. LightGBM must be
                    given at least what the benchmark has, or a win would only
                    prove it had better inputs.
RV lags             hours t-1 .. t-24, so the model can find its own decay
                    shape rather than being handed Corsi's.
Semivariance        RV split by sign of the minute return. The leverage effect
                    says downside moves raise future volatility more than
                    upside ones, which a symmetric RV cannot express.
Range estimators    Parkinson and Garman-Klass use the high and low, so they
                    see intrabar movement that close-to-close returns miss.
Activity            trades, volume, and taker-buy share. Order-flow imbalance
                    is informative and is already in the raw bars.
Calendar            hour of day and day of week. Recon measured a 1.29x
                    intraday volatility spread and calmer weekends.
Returns             recent signed returns, again for the leverage effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BARS_PER_HOUR = 55


def _hourly_base(bars: pd.DataFrame) -> pd.DataFrame:
    """Collapse 1-minute bars into one row per hour, gap-aware."""
    b = bars.set_index("ts").copy()
    r = np.log(b["close"] / b["close"].shift(1))
    # a return spanning a missing minute is not a 1-minute return
    r = r.where(b.index.to_series().diff() == pd.Timedelta("1min"))
    b["r"] = r
    b["r2"] = r ** 2
    b["r2_pos"] = np.where(r > 0, r ** 2, 0.0)
    b["r2_neg"] = np.where(r < 0, r ** 2, 0.0)
    # Parkinson and Garman-Klass need positive ranges
    hl = np.log(b["high"] / b["low"])
    co = np.log(b["close"] / b["open"])
    b["park"] = hl ** 2 / (4 * np.log(2))
    b["gk"] = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2

    g = b.groupby(pd.Grouper(freq="h"))
    h = pd.DataFrame({
        "rv": np.sqrt(g["r2"].sum()),
        "rv_pos": np.sqrt(g["r2_pos"].sum()),
        "rv_neg": np.sqrt(g["r2_neg"].sum()),
        "parkinson": np.sqrt(g["park"].sum()),
        "garman_klass": np.sqrt(g["gk"].sum().clip(lower=0)),
        "ret": g["r"].sum(),
        "trades": g["trades"].sum(),
        "volume": g["volume"].sum(),
        "taker_share": g["taker_buy_base"].sum() / g["volume"].sum().replace(0, np.nan),
        "n": g["r"].count(),
    })
    # an hour missing more than five minutes is not a clean RV; an outage and a
    # calm hour are indistinguishable unless you check
    return h[h["n"] >= MIN_BARS_PER_HOUR].drop(columns="n")


def build(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a frame indexed by hour t with feature columns and column `y` =
    log RV of hour t+1."""
    h = _hourly_base(bars)
    log_rv = np.log(h["rv"].replace(0, np.nan))

    f = pd.DataFrame(index=h.index)

    # --- HAR components (what the benchmark gets) ---------------------------
    f["har_h"] = log_rv
    f["har_d"] = log_rv.rolling(24).mean()
    f["har_w"] = log_rv.rolling(168).mean()

    # --- RV lags ------------------------------------------------------------
    for L in (1, 2, 3, 4, 6, 12, 24):
        f[f"log_rv_lag{L}"] = log_rv.shift(L)

    # --- dispersion of recent volatility ------------------------------------
    f["log_rv_std24"] = log_rv.rolling(24).std()
    f["log_rv_min24"] = log_rv.rolling(24).min()
    f["log_rv_max24"] = log_rv.rolling(24).max()

    # --- asymmetry / leverage -----------------------------------------------
    # An hour can contain almost no upward minute moves, which sends rv_pos to
    # near zero. A plain rv_neg/rv_pos ratio is therefore UNBOUNDED: measured
    # on this archive it reaches 9.3 in train but 1,929 in test, a 209x escape
    # from the fitted range. Scaled linear models and MLPs extrapolate on that
    # and blow up (test RMSE 1.8 and 3.4 against 0.33 in validation); trees are
    # immune only because they cannot extrapolate at all.
    #
    # So the ratio is replaced by the downside SHARE of variance, which carries
    # the same leverage information and is bounded to [0, 1] by construction.
    rv2_pos, rv2_neg = h["rv_pos"] ** 2, h["rv_neg"] ** 2
    f["downside_share"] = rv2_neg / (rv2_neg + rv2_pos).replace(0, np.nan)

    # Log-volatility floors: an hour with essentially no movement on one side
    # drives these to -15 and beyond, far outside anything seen in training.
    # Flooring keeps a rare, real observation from acting as an outlier.
    LOG_FLOOR = -10.0
    f["log_rv_pos"] = np.log(h["rv_pos"].replace(0, np.nan)).clip(lower=LOG_FLOOR)
    f["log_rv_neg"] = np.log(h["rv_neg"].replace(0, np.nan)).clip(lower=LOG_FLOOR)

    # --- range-based estimators ---------------------------------------------
    f["log_parkinson"] = np.log(h["parkinson"].replace(0, np.nan))
    f["log_garman_klass"] = np.log(h["garman_klass"].replace(0, np.nan))

    # --- signed returns ------------------------------------------------------
    f["ret_1h"] = h["ret"]
    f["ret_24h"] = h["ret"].rolling(24).sum()
    f["ret_1h_neg"] = h["ret"].clip(upper=0)

    # --- activity ------------------------------------------------------------
    f["log_trades"] = np.log1p(h["trades"])
    f["log_volume"] = np.log1p(h["volume"])
    f["taker_share"] = h["taker_share"]
    f["log_trades_chg"] = np.log1p(h["trades"]) - np.log1p(h["trades"]).rolling(24).mean()

    # --- calendar ------------------------------------------------------------
    f["hour"] = h.index.hour
    f["dow"] = h.index.dayofweek

    # --- label: next hour's log RV -------------------------------------------
    f["y"] = log_rv.shift(-1)

    # Only keep rows whose label is genuinely the NEXT hour. After dropping
    # unclean hours the index has holes, so shift(-1) can silently pair hour t
    # with an hour two or more steps later.
    next_is_adjacent = (
        pd.Series(f.index, index=f.index).shift(-1) - pd.Series(f.index, index=f.index)
    ) == pd.Timedelta("1h")
    f = f[next_is_adjacent.fillna(False)]

    return f.dropna()


FEATURES = None  # set on first build; see train script
