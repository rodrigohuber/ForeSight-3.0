"""
Diebold-Mariano test for equal predictive accuracy.

An RMSE that is lower is not automatically *better*: the difference has to be
larger than the noise in the loss differential. DM is the standard test for
this, and it is what turns "LightGBM beat HAR by 14.8%" from an observation
into a claim.

Newey-West (HAC) standard errors are used because forecast-error differentials
are serially correlated -- volatility forecasts miss in runs, not independently.
Ignoring that autocorrelation understates the variance and manufactures
significance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    *, lag: int | None = None) -> dict[str, float]:
    """Test whether A forecasts better than B, on squared-error loss.

    Returns the DM statistic and a two-sided p-value. A negative statistic
    means A has the lower loss, i.e. A is the better forecaster.
    """
    d = (y - pred_a) ** 2 - (y - pred_b) ** 2
    n = len(d)
    dbar = float(np.mean(d))

    # Newey-West long-run variance with Bartlett kernel
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - dbar
    gamma0 = float(dc @ dc) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        gk = float(dc[k:] @ dc[:-k]) / n
        lrv += 2 * (1 - k / (lag + 1)) * gk
    lrv = max(lrv, 1e-300)

    stat = dbar / np.sqrt(lrv / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"dm_stat": float(stat), "p_value": float(p),
            "mean_loss_diff": dbar, "hac_lag": float(lag), "n": float(n)}


def verdict(res: dict[str, float], a: str, b: str, alpha: float = 0.01) -> str:
    if res["p_value"] >= alpha:
        return f"{a} vs {b}: no significant difference (p={res['p_value']:.3f})"
    better = a if res["dm_stat"] < 0 else b
    return (f"{a} vs {b}: {better} is better, "
            f"DM={res['dm_stat']:+.2f}, p={res['p_value']:.2e}")
