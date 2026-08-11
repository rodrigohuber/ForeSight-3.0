"""
The chartist's toolkit, quantified.

Every indicator a technical analyst reads off a chart, computed exactly so it
can be tested rather than argued about. Nothing here is invented: these are the
standard formulas, with the standard parameters, as used on trading platforms.

What the evidence says, before we add to it
-------------------------------------------
Technical analysis splits into two things that deserve very different
treatment. The part that formalises cleanly -- trend, momentum, volatility
bands, volume confirmation -- has genuine published support: Brock, Lakonishok
& LeBaron (1992) for moving averages, and time-series momentum (Moskowitz, Ooi
& Pedersen 2012) which IS a moving-average rule in disguise. The part that
needs human judgement -- head-and-shoulders, Elliott waves, Fibonacci
retracements -- resists specification, and when Lo, Mamaysky & Wang (2000)
formalised patterns properly with kernel regression they found effects that
were statistically detectable but economically small.

Park & Irwin's (2007) survey of 95 studies found 56 positive, but flagged data
snooping, ex-post rule selection, and missing transaction costs as pervasive.
This module exists so those three failures can be avoided: the rules are fixed
in advance, evaluation is walk-forward, and costs are charged.

Note that our volatility model already contained much of this. har_d/har_w/
har_m are moving averages, ret_* are momentum, log_rv_std24 is a Bollinger
width, and Parkinson is an ATR relative. What is new here is applying the
toolkit to DIRECTION, which is the use chartists actually claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing, used by RSI/ATR/ADX -- not a simple EMA."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _rma(d.clip(lower=0), n)
    loss = _rma((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    line = close.ewm(span=fast, adjust=False).mean() - \
           close.ewm(span=slow, adjust=False).mean()
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = ma + k * sd, ma - k * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / ma.replace(0, np.nan)
    return pct_b, width


def atr(high, low, close, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return _rma(tr, n)


def adx(high, low, close, n: int = 14):
    up, dn = high.diff(), -low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(high, low, close, n).replace(0, np.nan)
    pdi = 100 * _rma(pd.Series(plus, index=high.index), n) / a
    mdi = 100 * _rma(pd.Series(minus, index=high.index), n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _rma(dx, n), pdi, mdi


def stochastic(high, low, close, n: int = 14, d: int = 3):
    hh, ll = high.rolling(n).max(), low.rolling(n).min()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def build(bars: pd.DataFrame, rule: str = "D") -> pd.DataFrame:
    """Resample to `rule` bars and compute the standard indicator set.

    Every column is strictly lagged: the row at t uses only bars up to and
    including t, and the label is the return of bar t+1.
    """
    b = bars.set_index("ts")
    o = b["open"].resample(rule).first()
    h = b["high"].resample(rule).max()
    l = b["low"].resample(rule).min()
    c = b["close"].resample(rule).last()
    v = b["volume"].resample(rule).sum()
    n_bars = b["close"].resample(rule).count()
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    df = df[n_bars > 0].dropna()

    f = pd.DataFrame(index=df.index)
    close = df["close"]

    # --- trend: moving averages and their crossings ------------------------
    for n in (10, 20, 50, 100, 200):
        f[f"px_vs_sma{n}"] = close / close.rolling(n).mean() - 1
    f["sma20_vs_50"] = (close.rolling(20).mean() /
                        close.rolling(50).mean() - 1)
    f["golden_cross"] = (close.rolling(50).mean() /
                         close.rolling(200).mean() - 1)     # >0 = golden

    # --- momentum / rate of change ------------------------------------------
    for n in (1, 5, 10, 20, 60, 120, 250):
        f[f"roc{n}"] = close.pct_change(n)
    f["rsi14"] = rsi(close, 14)
    f["rsi14_centered"] = f["rsi14"] - 50
    line, sig, hist = macd(close)
    f["macd_hist"] = hist / close                      # scale-free
    f["macd_cross"] = np.sign(hist)

    # --- volatility bands ----------------------------------------------------
    pct_b, width = bollinger(close)
    f["boll_pct_b"] = pct_b
    f["boll_width"] = width
    a = atr(df["high"], df["low"], close)
    f["atr_rel"] = a / close
    f["atr_chg"] = a / a.rolling(20).mean() - 1

    # --- trend strength ------------------------------------------------------
    adx_v, pdi, mdi = adx(df["high"], df["low"], close)
    f["adx"] = adx_v
    f["di_spread"] = pdi - mdi

    # --- oscillators ---------------------------------------------------------
    k, d = stochastic(df["high"], df["low"], close)
    f["stoch_k"] = k
    f["stoch_kd"] = k - d

    # --- support / resistance (Donchian distance) ----------------------------
    for n in (20, 50, 100):
        hh, ll = close.rolling(n).max(), close.rolling(n).min()
        f[f"dist_high{n}"] = close / hh - 1              # 0 = at the high
        f[f"dist_low{n}"] = close / ll - 1
    f["donchian_pos20"] = ((close - close.rolling(20).min()) /
                           (close.rolling(20).max() -
                            close.rolling(20).min()).replace(0, np.nan))

    # --- candle anatomy ------------------------------------------------------
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_frac"] = (df["close"] - df["open"]) / rng
    f["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    # --- volume confirmation -------------------------------------------------
    f["vol_ratio"] = v / v.rolling(20).mean()
    obv = (np.sign(close.diff()) * v).fillna(0).cumsum()
    f["obv_slope"] = obv.diff(10) / v.rolling(20).mean().replace(0, np.nan)

    # --- label: NEXT bar's log return ---------------------------------------
    f["y"] = np.log(close.shift(-1) / close)
    f["y_dir"] = (f["y"] > 0).astype(int)

    return f.replace([np.inf, -np.inf], np.nan).dropna()
