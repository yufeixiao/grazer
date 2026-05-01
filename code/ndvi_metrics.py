"""
Output-error metrics in NDVI space.

Spec §3.1. The reduced two-state recovery model is a deterministic ODE
under the OEM-MLE generative model: process noise is zero, observation
noise is iid Gaussian. Parameters are fit by minimizing SSE between
observed NDVI and an open-loop simulation. AIC follows from the
concentrated Gaussian likelihood.

Public API:
  simulate_open_loop(t, dose, params, model)  -> (y_hat, h, d)
  sse(y, y_hat)                                -> float
  ndvi_rmse(y, y_hat), ndvi_nse(y, y_hat)      -> float
  aic_oem(sse_value, n, k)                     -> float
  nll_oem(sse_value, n)                        -> float
  ljungbox_residual(y, y_hat, n_lags=12)       -> (Q, p)
"""
from __future__ import annotations

import numpy as np

from .detection import ljung_box


def simulate_open_loop(t_idx: np.ndarray,
                       dose: np.ndarray,
                       params: dict,
                       model: str = "ekf") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward integrate the deterministic recovery dynamics on `t_idx`.

    Operates on the anomaly series (Stage 0 has already removed the seasonal
    cycle). The observation model is ã_k = h_k, so y_hat = h directly.

    For `model='one'` the d-channel is held at zero and kappa is ignored.
    """
    t = np.asarray(t_idx, dtype=float)
    s = np.asarray(dose, dtype=float)
    T = t.size
    dt = 1.0

    r     = float(params["r"])
    h0    = float(params["h0"])
    gamma = float(params.get("gamma", 0.6))

    if model == "one":
        kappa = 0.0
        a     = 0.0
        tauD  = 1.0
        d0    = 0.0
    else:
        kappa = float(params.get("kappa", 0.0))
        a     = float(params.get("a", 0.8))
        tauD  = max(float(params.get("tauD", 1.0)), 1e-3)
        d0    = float(params.get("d0", 0.0))

    h = np.empty(T); d = np.empty(T)
    h[0] = h0
    d[0] = d0
    for k in range(T - 1):
        s_k = s[k]
        surv = np.exp(-dt * gamma * s_k)
        h[k+1] = (1.0 - dt*r) * surv * h[k] - dt*kappa*d[k]*h[k] + dt*r*h0
        d[k+1] = (1.0 - dt/tauD) * d[k] + a * dt * s_k

    y_hat = h.copy()
    return y_hat, h, d


def _finite_pair(y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(y_hat)
    return y[mask], y_hat[mask]


def sse(y: np.ndarray, y_hat: np.ndarray) -> float:
    a, b = _finite_pair(y, y_hat)
    if a.size == 0:
        return float("inf")
    r = a - b
    return float(np.dot(r, r))


def ndvi_rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    a, b = _finite_pair(y, y_hat)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ndvi_nse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency. 1 is perfect, 0 is mean baseline,
    negative is worse than mean."""
    a, b = _finite_pair(y, y_hat)
    if a.size == 0:
        return float("nan")
    denom = float(np.sum((a - a.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / denom)


def nll_oem(sse_value: float, n: int) -> float:
    """Negative log-likelihood under iid Gaussian observation noise with
    sigma^2 concentrated to SSE/n. Spec §3.1.

        -2 log L = n log(SSE/n) + n (1 + log 2 pi)
    """
    if n <= 0 or not np.isfinite(sse_value) or sse_value <= 0:
        return float("inf")
    return 0.5 * (n * np.log(sse_value / n) + n * (1.0 + np.log(2*np.pi)))


def aic_oem(sse_value: float, n: int, k: int) -> float:
    """AIC = 2k + 2 * nll. Equivalent to n log(SSE/n) + 2k up to a
    constant that cancels in ΔAIC."""
    return 2.0 * k + 2.0 * nll_oem(sse_value, n)


def ljungbox_residual(y: np.ndarray, y_hat: np.ndarray, n_lags: int = 12) -> tuple[float, float]:
    """Ljung-Box on the centered, standardized residual r_k = y_k - ŷ_k.
    Returns (Q, p)."""
    a, b = _finite_pair(y, y_hat)
    if a.size <= n_lags + 1:
        return float("nan"), float("nan")
    r = a - b
    s = float(np.std(r))
    z = (r - r.mean()) / s if s > 0 else r - r.mean()
    return ljung_box(z, n_lags=n_lags)
