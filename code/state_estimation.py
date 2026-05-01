"""
Stage 2 state estimation. EKF and RTS smoother at fixed parameters.

Spec §3.2. After OEM-MLE fixes θ̂, the filter runs as a state estimator
only. Process noise Q is a small numerical regularizer, set rather than
fit. Observation noise R is set from the OEM σ̂². The filter outputs
the standardized innovations consumed by Stage 3 detection.

Public API:
  ekf_smooth(t, y, dose, params, q_floor=1e-6) -> dict
  cusum_trace(z, drift=0.5, threshold=5.0)     -> dict
"""
from __future__ import annotations

import numpy as np


def _ekf_forward(t_idx, y, dose, params, q_floor):
    dt = 1.0
    r     = params["r"]
    kappa = params.get("kappa", 0.0)
    tauD  = max(params.get("tauD", 1e6), 1e-3)
    gamma = params.get("gamma", 0.6)
    a     = params.get("a", 0.8)
    h0    = params["h0"]
    d0    = params.get("d0", 0.0)
    R     = float(params.get("R", 1e-3))

    Q = np.diag([q_floor, q_floor])
    T = len(y)
    x_pred = np.zeros((T, 2)); P_pred = np.zeros((T, 2, 2))
    x_filt = np.zeros((T, 2)); P_filt = np.zeros((T, 2, 2))
    F_hist = np.zeros((T, 2, 2))
    y_pred = np.full(T, np.nan)
    S_arr  = np.full(T, np.nan)
    nu_arr = np.full(T, np.nan)
    z_arr  = np.full(T, np.nan)

    x = np.array([h0, d0])
    P = np.diag([0.01, 0.05])

    for k in range(T):
        s_k = dose[k]
        if k == 0:
            x_p = x.copy()
            P_p = P.copy()
            F = np.eye(2)
        else:
            h, d = x[0], x[1]
            surv = np.exp(-dt * gamma * s_k_prev)
            f0 = (1.0 - dt*r) * surv * h - dt*kappa*d*h + dt*r*h0
            f1 = (1.0 - dt/tauD) * d + a * dt * s_k_prev
            F = np.array([
                [(1.0 - dt*r) * surv - dt*kappa*d, -dt*kappa*h],
                [0.0,                              1.0 - dt/tauD],
            ])
            x_p = np.array([f0, f1])
            P_p = F @ P @ F.T + Q

        x_pred[k] = x_p; P_pred[k] = P_p; F_hist[k] = F

        C = np.array([1.0, 0.0])
        if np.isfinite(y[k]):
            y_hat = C @ x_p
            S = float(C @ P_p @ C + R)
            nu = float(y[k] - y_hat)
            K = (P_p @ C) / S
            x = x_p + K * nu
            P = P_p - np.outer(K, C) @ P_p
            y_pred[k] = y_hat
            S_arr[k]  = S
            nu_arr[k] = nu
            z_arr[k]  = nu / np.sqrt(S) if S > 0 else np.nan
        else:
            x = x_p
            P = P_p
        x_filt[k] = x; P_filt[k] = P
        s_k_prev = s_k

    return {
        "x_pred": x_pred, "P_pred": P_pred,
        "x_filt": x_filt, "P_filt": P_filt,
        "F":      F_hist,
        "y_pred": y_pred, "S": S_arr, "nu": nu_arr, "z": z_arr,
    }


def _rts_backward(fwd):
    T = fwd["x_filt"].shape[0]
    x_smooth = fwd["x_filt"].copy()
    P_smooth = fwd["P_filt"].copy()
    for k in range(T - 2, -1, -1):
        # P_pred at k+1 was computed from F at k+1 applied to P_filt at k.
        F_next = fwd["F"][k+1]
        P_pred_next = fwd["P_pred"][k+1]
        try:
            G = fwd["P_filt"][k] @ F_next.T @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            continue
        x_smooth[k] = fwd["x_filt"][k] + G @ (x_smooth[k+1] - fwd["x_pred"][k+1])
        P_smooth[k] = fwd["P_filt"][k] + G @ (P_smooth[k+1] - P_pred_next) @ G.T
    return x_smooth, P_smooth


def ekf_smooth(t_idx, y, dose, params, q_floor: float = 1e-6) -> dict:
    """EKF forward + RTS backward at fixed parameters.

    Returns:
      x_filt, P_filt, x_smooth, P_smooth, y_pred, S, nu, z
    where z is the standardized innovation `nu / sqrt(S)`.
    """
    t_idx = np.asarray(t_idx, dtype=float)
    y     = np.asarray(y, dtype=float)
    dose  = np.asarray(dose, dtype=float)
    fwd = _ekf_forward(t_idx, y, dose, params, q_floor)
    x_smooth, P_smooth = _rts_backward(fwd)
    return {
        "t":        t_idx,
        "x_filt":   fwd["x_filt"],
        "P_filt":   fwd["P_filt"],
        "x_smooth": x_smooth,
        "P_smooth": P_smooth,
        "y_pred":   fwd["y_pred"],
        "S":        fwd["S"],
        "nu":       fwd["nu"],
        "z":        fwd["z"],
    }


def cusum_trace(z: np.ndarray, drift: float = 0.5,
                threshold: float = 5.0) -> dict:
    """Two-sided CUSUM on standardized innovations. Spec §3.3.

    g_k^+ = max(0, g_{k-1}^+ + z_k - drift)
    g_k^- = min(0, g_{k-1}^- + z_k + drift)
    Alarms when |g_k| > threshold.
    """
    z = np.asarray(z, dtype=float)
    z_clean = np.where(np.isfinite(z), z, 0.0)
    n = z_clean.size
    g_pos = np.zeros(n); g_neg = np.zeros(n)
    for k in range(1, n):
        g_pos[k] = max(0.0, g_pos[k-1] + z_clean[k] - drift)
        g_neg[k] = min(0.0, g_neg[k-1] + z_clean[k] + drift)
    alarm_pos = np.where(g_pos >  threshold)[0]
    alarm_neg = np.where(g_neg < -threshold)[0]
    return {
        "g_pos":     g_pos,
        "g_neg":     g_neg,
        "alarm_pos": alarm_pos,
        "alarm_neg": alarm_neg,
        "threshold": threshold,
        "drift":     drift,
    }
