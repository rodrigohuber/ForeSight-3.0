"""
Feature set v2 — jump-robust, measurement-error aware, multi-horizon.

Phase 1 showed LightGBM beating HAR by 11% out of sample but badly missing the
tail (6.6% realized against a 1.2% forecast). v2 targets that directly, using
the estimators the realized-volatility literature developed for exactly this
problem.

What v2 adds over v1
--------------------
Bipower variation   BV = (pi/2) * sum |r_i| |r_{i-1}|. Robust to jumps: it
                    estimates the CONTINUOUS part of variation, because a lone
                    huge return has no equally huge neighbour to pair with.
Jump component      J = max(RV - BV, 0). Andersen-Bollerslev-Diebold showed the
                    continuous and jump parts have very different persistence --
                    jumps barely predict future volatility while the continuous
                    part strongly does. HAR cannot separate them.
Realized quarticity RQ = (n/3) * sum r_i^4, the variance of the RV estimator
                    itself. HARQ (Bollerslev-Patton-Quaedvlieg 2016) uses it to
                    downweight hours whose RV is measured noisily. This is the
                    single most cited improvement to HAR.
Longer memory       a monthly (720h) component; v1 stopped at a week.
Seasonal adjustment RV divided by its hour-of-week average, so the model sees
                    deviation from the normal level for that time slot rather
                    than re-learning the 1.29x intraday pattern in every tree.
Tail context        how far the recent maximum sits above the recent median --
                    a spike-proximity signal the mean-focused features lack.

Every feature remains strictly lagged. The label for hour t is volatility over
the NEXT horizon, and horizons are configurable (1h, 4h, 24h).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BARS_PER_HOUR = 55
LOG_FLOOR = -10.0


def hourly_base(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.set_index("ts").copy()
    r = np.log(b["close"] / b["close"].shift(1))
    r = r.where(b.index.to_series().diff() == pd.Timedelta("1min"))
    b["r"] = r
    b["r2"] = r ** 2
    b["r4"] = r ** 4
    b["absr"] = r.abs()
    # bipower needs |r_t| * |r_{t-1}|, and both must be real returns
    b["bp"] = b["absr"] * b["absr"].shift(1)
    b.loc[b["absr"].shift(1).isna(), "bp"] = np.nan
    b["r2_pos"] = np.where(r > 0, r ** 2, 0.0)
    b["r2_neg"] = np.where(r < 0, r ** 2, 0.0)
    hl = np.log(b["high"] / b["low"])
    co = np.log(b["close"] / b["open"])
    b["park"] = hl ** 2 / (4 * np.log(2))
    b["gk"] = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2

    g = b.groupby(pd.Grouper(freq="h"))
    n = g["r"].count()
    h = pd.DataFrame({
        "rv2": g["r2"].sum(),
        "bv2": (np.pi / 2) * g["bp"].sum(),
        "rq": (n / 3) * g["r4"].sum(),
        "rv_pos": np.sqrt(g["r2_pos"].sum()),
        "rv_neg": np.sqrt(g["r2_neg"].sum()),
        "parkinson": np.sqrt(g["park"].sum()),
        "garman_klass": np.sqrt(g["gk"].sum().clip(lower=0)),
        "ret": g["r"].sum(),
        "trades": g["trades"].sum(),
        "volume": g["volume"].sum(),
        "taker": g["taker_buy_base"].sum(),
        "n": n,
    })
    h = h[h["n"] >= MIN_BARS_PER_HOUR]
    h["rv"] = np.sqrt(h["rv2"])
    h["bv"] = np.sqrt(h["bv2"].clip(lower=0))
    h["jump"] = np.sqrt((h["rv2"] - h["bv2"]).clip(lower=0))
    return h


def _seasonal(log_rv: pd.Series) -> pd.Series:
    """Hour-of-week seasonal level, computed EXPANDING so it never uses the
    future. A seasonal index fitted on the whole sample is a subtle leak."""
    how = log_rv.index.dayofweek * 24 + log_rv.index.hour
    df = pd.DataFrame({"v": log_rv.to_numpy(), "how": how}, index=log_rv.index)
    # expanding mean per slot, shifted so hour t uses only earlier weeks
    return df.groupby("how")["v"].transform(
        lambda s: s.shift(1).expanding(min_periods=4).mean())


def build(bars: pd.DataFrame, horizon: int = 1,
          require_label: bool = True) -> pd.DataFrame:
    """Features at hour t; label = log volatility over the next `horizon` hours.

    `require_label=False` keeps the most recent hours, whose label lies in the
    future and is therefore NaN. Those rows are the only ones a genuine live
    forecast can be issued from: with require_label=True the newest usable row
    is `horizon` hours old, so predicting from it would re-predict a window
    already present in the data.
    """
    h = hourly_base(bars)
    log_rv = np.log(h["rv"].replace(0, np.nan))
    f = pd.DataFrame(index=h.index)

    # --- HAR family, extended to a monthly term -----------------------------
    f["har_h"] = log_rv
    f["har_d"] = log_rv.rolling(24).mean()
    f["har_w"] = log_rv.rolling(168).mean()
    f["har_m"] = log_rv.rolling(720).mean()

    # --- jump / continuous decomposition ------------------------------------
    log_bv = np.log(h["bv"].replace(0, np.nan)).clip(lower=LOG_FLOOR)
    f["log_bv"] = log_bv
    f["log_bv_d"] = log_bv.rolling(24).mean()
    f["jump_share"] = (h["rv2"] - h["bv2"]).clip(lower=0) / h["rv2"].replace(0, np.nan)
    f["jump_share_d"] = f["jump_share"].rolling(24).mean()

    # --- measurement error (HARQ) -------------------------------------------
    # RQ enters as sqrt(RQ) interacted with the hourly term, which is the HARQ
    # form: when this hour's RV is noisily measured, lean less on it.
    rq_scaled = np.sqrt(h["rq"].clip(lower=0))
    f["sqrt_rq"] = np.log1p(rq_scaled)
    f["harq_interact"] = f["sqrt_rq"] * log_rv

    # --- lags ----------------------------------------------------------------
    for L in (1, 2, 3, 4, 6, 12, 24, 48):
        f[f"log_rv_lag{L}"] = log_rv.shift(L)

    # --- dispersion and tail context ----------------------------------------
    f["log_rv_std24"] = log_rv.rolling(24).std()
    f["log_rv_std168"] = log_rv.rolling(168).std()
    f["log_rv_max24"] = log_rv.rolling(24).max()
    f["log_rv_min24"] = log_rv.rolling(24).min()
    med24 = log_rv.rolling(24).median()
    f["spike_gap"] = log_rv.rolling(24).max() - med24
    f["above_med"] = log_rv - med24
    f["trend_24_168"] = log_rv.rolling(24).mean() - log_rv.rolling(168).mean()

    # --- asymmetry / leverage (bounded; see v1 notes) ------------------------
    rv2p, rv2n = h["rv_pos"] ** 2, h["rv_neg"] ** 2
    f["downside_share"] = rv2n / (rv2n + rv2p).replace(0, np.nan)
    f["downside_share_d"] = f["downside_share"].rolling(24).mean()
    f["log_rv_neg"] = np.log(h["rv_neg"].replace(0, np.nan)).clip(lower=LOG_FLOOR)
    f["log_rv_pos"] = np.log(h["rv_pos"].replace(0, np.nan)).clip(lower=LOG_FLOOR)

    # --- range estimators ----------------------------------------------------
    f["log_parkinson"] = np.log(h["parkinson"].replace(0, np.nan)).clip(lower=LOG_FLOOR)
    f["log_garman_klass"] = np.log(h["garman_klass"].replace(0, np.nan)).clip(lower=LOG_FLOOR)
    f["park_minus_rv"] = f["log_parkinson"] - log_rv

    # --- returns / leverage --------------------------------------------------
    f["ret_1h"] = h["ret"]
    f["ret_6h"] = h["ret"].rolling(6).sum()
    f["ret_24h"] = h["ret"].rolling(24).sum()
    f["ret_1h_neg"] = h["ret"].clip(upper=0)
    f["ret_24h_neg"] = f["ret_24h"].clip(upper=0)

    # --- activity ------------------------------------------------------------
    lt = np.log1p(h["trades"])
    f["log_trades"] = lt
    f["log_trades_chg"] = lt - lt.rolling(24).mean()
    f["log_volume"] = np.log1p(h["volume"])
    f["taker_share"] = h["taker"] / h["volume"].replace(0, np.nan)
    f["taker_imbalance"] = (2 * f["taker_share"] - 1).abs()

    # --- seasonality (expanding, no lookahead) -------------------------------
    seas = _seasonal(log_rv)
    f["seasonal_level"] = seas
    f["rv_vs_seasonal"] = log_rv - seas

    # --- calendar ------------------------------------------------------------
    f["hour"] = h.index.hour
    f["dow"] = h.index.dayofweek

    # --- label: log volatility over the next `horizon` hours -----------------
    fwd = h["rv2"].rolling(horizon).sum().shift(-horizon)
    f["y"] = np.log(np.sqrt(fwd.replace(0, np.nan)))

    # the label must cover `horizon` CONTIGUOUS future hours
    idx = pd.Series(h.index, index=h.index)
    ok = (idx.shift(-horizon) - idx) == pd.Timedelta(hours=horizon)

    f = f.replace([np.inf, -np.inf], np.nan)
    if require_label:
        return f[ok.fillna(False)].dropna()

    # keep rows whose FEATURES are complete, label or not
    feat_cols = [c for c in f.columns if c != "y"]
    return f[f[feat_cols].notna().all(axis=1)]
