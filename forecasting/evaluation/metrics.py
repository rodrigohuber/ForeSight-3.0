"""
Forecast evaluation metrics.

Kept separate from any model so every model is scored by identical code. A
benchmark where each candidate brings its own metric is not a benchmark.
"""

from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    """QLIKE loss on the variance scale, y and yhat given as volatilities.

    QLIKE = mean( v/vhat - log(v/vhat) - 1 ), v = y^2.

    Preferred over MSE for volatility because it is robust to noise in the
    volatility proxy and penalises under-prediction more than over-prediction,
    which is the asymmetry a risk user actually cares about. Minimised at
    vhat = v, where it equals 0.
    """
    v, vhat = y ** 2, yhat ** 2
    ok = (v > 0) & (vhat > 0)
    ratio = v[ok] / vhat[ok]
    return float(np.mean(ratio - np.log(ratio) - 1))


def r2_vs(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray) -> float:
    """Out-of-sample R^2 against a stated baseline, not against the mean.

    Positive means the model beats that baseline; 0 means it ties it; negative
    means it is worse. Reporting R^2 against the in-sample mean flatters a
    forecast that has learned nothing but the average level.
    """
    ss_model = float(np.sum((y - yhat) ** 2))
    ss_base = float(np.sum((y - baseline) ** 2))
    return 1.0 - ss_model / ss_base if ss_base > 0 else float("nan")


def hit_rate(y: np.ndarray, yhat: np.ndarray) -> float:
    """Directional accuracy, ignoring bars where the truth is exactly zero.

    Those bars are a large point mass at minute granularity (2.86% of the
    corpus) and counting them as 'correct' inflates the score for free.
    """
    ok = y != 0
    return float(np.mean(np.sign(y[ok]) == np.sign(yhat[ok]))) if ok.any() else float("nan")


def summarize(y: np.ndarray, yhat: np.ndarray, baseline: np.ndarray,
              *, volatility: bool) -> dict[str, float]:
    out = {
        "rmse": rmse(y, yhat),
        "mae": mae(y, yhat),
        "r2_vs_baseline": r2_vs(y, yhat, baseline),
    }
    out["qlike"] = qlike(y, yhat) if volatility else float("nan")
    out["hit_rate"] = float("nan") if volatility else hit_rate(y, yhat)
    return out
