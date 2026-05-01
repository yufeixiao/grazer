"""
Kalman filter variants for the reduced NDVI recovery model.

Three filters, all monthly dt=1:
    kalman_filter_1state   — visible-only null (framework §8)
    kalman_filter_2state   — linear KF with d-state (framework §11, eqs 30-35)
    ekf_filter_2state      — EKF with nonlinear d*h coupling (framework §13)

Each returns (nll, x_hist, extras). `extras` carries per-step pre-update
innovation ν_k = y_k − ŷ_{k|k−1} and its variance S_k, plus the χ² gate
flag when `gate_outliers=True` (Puhm 2020 §2.2.4). Gating skips the
measurement update on rejection so the filter does not "learn" a single
outlier (e.g., the fire month itself as a known anchor).

Fitting, priors, I/O, and CLI live in `grazer.inference.fit` / `.io`.
"""

from __future__ import annotations

import numpy as np

# χ² (df=1) 1-α critical value at α = 0.01. scipy.stats.chi2.ppf(0.99, 1).
# Hard-coded so kalman.py stays scipy-free; detection.py uses scipy for the
# general case.
_CHI2_1_0P99 = 6.6348966010212145


def _chi2_1_crit(alpha: float) -> float:
    if alpha == 0.01:
        return _CHI2_1_0P99
    from scipy.stats import chi2
    return float(chi2.ppf(1.0 - alpha, df=1))


def kalman_filter_2state(y, t_idx, dose, params,
                         gate_outliers: bool = False,
                         gate_alpha: float = 0.01):
    """Linear KF with time-varying forcing (c1 pinned to 1).

    Exponential-loss form (stable for any dose):
       h_{k+1} = (1 - dt*r) * exp(-dt*gamma*s_k) * h_k - dt*kappa d_k + dt*r*h0 + w^h
       d_{k+1} = (1 - dt/tauD) * d_k + a * s_k * dt + w^d
                 (forward-Euler discretization of framework eq 7)
    """
    dt = 1.0
    r = params["r"]; kappa = params["kappa"]; tauD = params["tauD"]
    gamma = params["gamma"]; a = params["a"]
    h0 = params["h0"]; c0 = params["c0"]
    A1 = params["A1"]; A2 = params["A2"]
    q_h = params["q_h"]; q_d = params["q_d"]; R = params["R"]

    T = len(y)
    x = np.array([h0, 0.0])
    P = np.diag([0.01, 0.05])
    x_hist = np.zeros((T, 2))
    nu_arr = np.full(T, np.nan)
    S_arr = np.full(T, np.nan)
    y_pred_arr = np.full(T, np.nan)
    gated = np.zeros(T, dtype=bool)
    chi2_crit = _chi2_1_crit(gate_alpha) if gate_outliers else np.inf
    nll = 0.0
    for k in range(T):
        tk = t_idx[k]
        s_k = dose[k]
        loss = 1.0 - np.exp(-dt * gamma * s_k)
        A_mat = np.array([
            [(1.0 - dt * r) * (1.0 - loss), -dt * kappa],
            [0.0,                            1.0 - dt / tauD],
        ])
        b_vec = np.array([dt * r * h0, a * dt * s_k])
        Q = np.diag([q_h, q_d])
        x = A_mat @ x + b_vec
        P = A_mat @ P @ A_mat.T + Q

        m_t = c0 + A1 * np.sin(2 * np.pi * tk / 12.0) + A2 * np.cos(2 * np.pi * tk / 12.0)
        C_vec = np.array([1.0, 0.0])
        if not np.isnan(y[k]):
            y_pred = C_vec @ x + m_t
            S = C_vec @ P @ C_vec + R
            innov = y[k] - y_pred
            y_pred_arr[k] = y_pred; S_arr[k] = S; nu_arr[k] = innov
            T_chi2 = innov * innov / S
            if gate_outliers and T_chi2 > chi2_crit:
                gated[k] = True
            else:
                nll += 0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
                K = (P @ C_vec) / S
                x = x + K * innov
                P = P - np.outer(K, C_vec) @ P
        x_hist[k] = x
    extras = {"nu": nu_arr, "S": S_arr, "y_pred": y_pred_arr, "gated": gated}
    return nll, x_hist, extras


def kalman_filter_1state(y, t_idx, dose, params,
                         gate_outliers: bool = False,
                         gate_alpha: float = 0.01):
    """Visible-only (no d); fire forcing multiplies h."""
    dt = 1.0
    r = params["r"]; gamma = params["gamma"]
    h0 = params["h0"]; c0 = params["c0"]
    A1 = params["A1"]; A2 = params["A2"]
    q_h = params["q_h"]; R = params["R"]

    T = len(y)
    x = h0; P = 0.01
    x_hist = np.zeros(T)
    nu_arr = np.full(T, np.nan)
    S_arr = np.full(T, np.nan)
    y_pred_arr = np.full(T, np.nan)
    gated = np.zeros(T, dtype=bool)
    chi2_crit = _chi2_1_crit(gate_alpha) if gate_outliers else np.inf
    nll = 0.0
    for k in range(T):
        tk = t_idx[k]; s_k = dose[k]
        loss = 1.0 - np.exp(-dt * gamma * s_k)
        A = (1.0 - dt * r) * (1.0 - loss)
        x = A * x + dt * r * h0
        P = A * A * P + q_h
        m_t = c0 + A1 * np.sin(2 * np.pi * tk / 12.0) + A2 * np.cos(2 * np.pi * tk / 12.0)
        if not np.isnan(y[k]):
            y_pred = x + m_t
            S = P + R
            innov = y[k] - y_pred
            y_pred_arr[k] = y_pred; S_arr[k] = S; nu_arr[k] = innov
            T_chi2 = innov * innov / S
            if gate_outliers and T_chi2 > chi2_crit:
                gated[k] = True
            else:
                nll += 0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
                K = P / S
                x = x + K * innov
                P = (1 - K) * P
        x_hist[k] = x
    extras = {"nu": nu_arr, "S": S_arr, "y_pred": y_pred_arr, "gated": gated}
    return nll, np.column_stack([x_hist, np.zeros(T)]), extras


def ekf_filter_2state(y, t_idx, dose, params,
                      gate_outliers: bool = False,
                      gate_alpha: float = 0.01):
    """EKF with nonlinear d*h transition (framework §13, eqs 46-51)."""
    dt = 1.0
    r = params["r"]; kappa = params["kappa"]; tauD = params["tauD"]
    gamma = params["gamma"]; a = params["a"]
    h0 = params["h0"]; c0 = params["c0"]
    A1 = params["A1"]; A2 = params["A2"]
    q_h = params["q_h"]; q_d = params["q_d"]; R = params["R"]

    T = len(y)
    x = np.array([h0, 0.0])
    P = np.diag([0.01, 0.05])
    x_hist = np.zeros((T, 2))
    nu_arr = np.full(T, np.nan)
    S_arr = np.full(T, np.nan)
    y_pred_arr = np.full(T, np.nan)
    gated = np.zeros(T, dtype=bool)
    chi2_crit = _chi2_1_crit(gate_alpha) if gate_outliers else np.inf
    nll = 0.0
    for k in range(T):
        tk = t_idx[k]; s_k = dose[k]
        h, d = x[0], x[1]
        loss = 1.0 - np.exp(-dt * gamma * s_k)
        surv = 1.0 - loss
        f0 = (1.0 - dt * r) * surv * h - dt * kappa * d * h + dt * r * h0
        f1 = (1.0 - dt / tauD) * d + a * dt * s_k
        F = np.array([
            [(1.0 - dt * r) * surv - dt * kappa * d, -dt * kappa * h],
            [0.0,                                     1.0 - dt / tauD],
        ])
        x = np.array([f0, f1])
        Q = np.diag([q_h, q_d])
        P = F @ P @ F.T + Q

        m_t = c0 + A1 * np.sin(2 * np.pi * tk / 12.0) + A2 * np.cos(2 * np.pi * tk / 12.0)
        C_vec = np.array([1.0, 0.0])
        if not np.isnan(y[k]):
            y_pred = C_vec @ x + m_t
            S = C_vec @ P @ C_vec + R
            innov = y[k] - y_pred
            y_pred_arr[k] = y_pred; S_arr[k] = S; nu_arr[k] = innov
            T_chi2 = innov * innov / S
            if gate_outliers and T_chi2 > chi2_crit:
                gated[k] = True
            else:
                nll += 0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
                K = (P @ C_vec) / S
                x = x + K * innov
                P = P - np.outer(K, C_vec) @ P
        x_hist[k] = x
    extras = {"nu": nu_arr, "S": S_arr, "y_pred": y_pred_arr, "gated": gated}
    return nll, x_hist, extras
