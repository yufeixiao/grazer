"""
Reduced nonlinear forward simulator (framework §9 eq. 24 + §10 eq. 27).

Discrete-time (monthly dt=1) Monte Carlo of the coupled H/D state:
    h_{k+1} = (1 - dt*r) * exp(-dt*gamma*s_k) * h_k
              - dt*kappa*d_k*h_k + dt*r*h_0 + eta_h
    d_{k+1} = (1 - dt/tau_D) * d_k + a * s_k * dt + eta_d
              (forward-Euler discretization of framework eq 7)
    y_k     = c_0 + c_1*h_k + c_2*d_k + A1 sin(2πt/12) + A2 cos(2πt/12) + ε_k

import `simulate_reduced_nonlinear` to use the core in notebooks or tests.

Phase 2 of the spec will add an `s_fn`-based continuous simulator with RK4
integration; the dose-array forward loop below is the discrete variant used
for MTBS-driven simulations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate(s_fn: Callable[[float], float],
             params: dict,
             t_grid: np.ndarray,
             x0: tuple[float, float] | None = None) -> dict:
    """RK4 integration of the continuous coupled H/D equations.

        dH/dt = (H0 - H)/tauH - gamma * s(t) * P * H - kappa * D * H
        dD/dt = a * s(t) * P - D/tauD

    Parameters
    ----------
    s_fn : callable t -> float
        Forcing protocol. Evaluated at midpoint and endpoints per RK4.
    params : dict
        Keys: tauH (or r=1/tauH), tauD, gamma, a, kappa, h0, P (default 1.0).
    t_grid : np.ndarray
        Monotone time grid. Δt need not be uniform.
    x0 : (H0, D0), optional
        Initial state. Defaults to (params['h0'], 0.0).

    Returns
    -------
    dict with keys H (N,), D (N,), t (N,).
    """
    if "tauH" in params:
        tauH = params["tauH"]
    elif "r" in params:
        tauH = 1.0 / params["r"]
    else:
        raise KeyError("params must supply tauH or r")
    tauD  = params["tauD"]
    gamma = params.get("gamma", 0.6)
    a     = params.get("a", 0.8)
    kappa = params.get("kappa", 0.0)
    h0    = params["h0"]
    P     = params.get("P", 1.0)

    t = np.asarray(t_grid, dtype=float)
    N = t.size
    H = np.empty(N); D = np.empty(N)
    if x0 is None:
        H[0], D[0] = h0, 0.0
    else:
        H[0], D[0] = x0

    def f(ti, Hi, Di):
        s = float(s_fn(ti))
        dH = (h0 - Hi) / tauH - gamma * s * P * Hi - kappa * Di * Hi
        dD = a * s * P - Di / tauD
        return dH, dD

    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        tk = t[k]; Hk = H[k]; Dk = D[k]
        k1H, k1D = f(tk,           Hk,                Dk)
        k2H, k2D = f(tk + 0.5*dt,  Hk + 0.5*dt*k1H,   Dk + 0.5*dt*k1D)
        k3H, k3D = f(tk + 0.5*dt,  Hk + 0.5*dt*k2H,   Dk + 0.5*dt*k2D)
        k4H, k4D = f(tk + dt,      Hk + dt*k3H,       Dk + dt*k3D)
        H[k + 1] = Hk + (dt / 6.0) * (k1H + 2*k2H + 2*k3H + k4H)
        D[k + 1] = Dk + (dt / 6.0) * (k1D + 2*k2D + 2*k3D + k4D)

    return {"t": t, "H": H, "D": D}


def simulate_reduced_nonlinear(t_idx: np.ndarray,
                               dose: np.ndarray,
                               params: dict,
                               rng: np.random.Generator,
                               c2: float = 0.0) -> dict:
    """Integrate the discrete-time nonlinear reduced model forward.

    Returns dict with h, d, y (synthetic NDVI w/ obs noise), y_mean
    (deterministic observation), m_t (seasonal mean).
    """
    dt = 1.0
    r      = params["r"]
    kappa  = params["kappa"]
    tauD   = params["tauD"]
    gamma  = params.get("gamma", 0.6)
    a      = params.get("a",     0.8)
    h0     = params["h0"]
    c0     = params.get("c0", 0.0)
    A1     = params.get("A1", 0.0)
    A2     = params.get("A2", 0.0)
    q_h    = params.get("q_h", 0.0)
    q_d    = params.get("q_d", 0.0)
    R      = params.get("R",   0.0)

    T = len(t_idx)
    h = np.empty(T); d = np.empty(T)
    h[0] = h0; d[0] = 0.0

    for k in range(T - 1):
        s_k = dose[k]
        loss = 1.0 - np.exp(-dt * gamma * s_k)
        surv = 1.0 - loss
        h_next = ((1.0 - dt * r) * surv * h[k]
                  - dt * kappa * d[k] * h[k]
                  + dt * r * h0)
        d_next = (1.0 - dt / tauD) * d[k] + a * dt * s_k
        h[k + 1] = h_next + np.sqrt(q_h) * rng.standard_normal() if q_h > 0 else h_next
        d[k + 1] = d_next + np.sqrt(q_d) * rng.standard_normal() if q_d > 0 else d_next

    m_t = c0 + A1 * np.sin(2 * np.pi * t_idx / 12.0) + A2 * np.cos(2 * np.pi * t_idx / 12.0)
    y_mean = m_t + h + c2 * d
    obs_noise = np.sqrt(max(R, 0.0)) * rng.standard_normal(T) if R > 0 else 0.0
    y_t = y_mean + obs_noise

    return {"h": h, "d": d, "y": y_t, "y_mean": y_mean, "m_t": m_t}


def plot_reduced_simulation(t_idx, y_obs, dose, ensemble, params, site_id, out_path):
    """Four-panel figure: NDVI (obs + ensemble), h(t), d(t), dose bars."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1, 0.7]})

    hs = np.stack([e["h"] for e in ensemble])
    ds = np.stack([e["d"] for e in ensemble])
    y_mean = ensemble[0]["y_mean"]

    ax = axes[0]
    ax.plot(t_idx, y_obs, "o", ms=3.5, color="0.25", alpha=0.85,
            label=f"Observed NDVI (monthly bin, {site_id})")
    for k, e in enumerate(ensemble):
        ax.plot(t_idx, e["y"], "-", color="C1", lw=0.6, alpha=0.25,
                label="Synthetic NDVI (stochastic)" if k == 0 else None)
    ax.plot(t_idx, y_mean, "-", color="C1", lw=2.0,
            label="Synthetic NDVI (mean model prediction)")
    ax.axvline(0, color="r", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("NDVI"); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Reduced nonlinear model forward sim — {site_id}  "
                 f"(n_seeds={len(ensemble)})")

    ax = axes[1]
    h_mean = hs.mean(axis=0); h_std = hs.std(axis=0)
    ax.plot(t_idx, h_mean, "-", color="C2", lw=1.8, label=r"$\bar h(t)$ ensemble mean")
    ax.fill_between(t_idx, h_mean - h_std, h_mean + h_std, color="C2", alpha=0.20,
                    label=r"$\pm 1\sigma$")
    ax.axhline(params["h0"], color="C2", ls=":", lw=1, alpha=0.7, label=r"$h_0$ baseline")
    ax.axvline(0, color="r", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel(r"$h_t$ (visible)"); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    d_mean = ds.mean(axis=0); d_std = ds.std(axis=0)
    ax.plot(t_idx, d_mean, "-", color="C3", lw=1.8, label=r"$\bar d(t)$ ensemble mean")
    ax.fill_between(t_idx, d_mean - d_std, d_mean + d_std, color="C3", alpha=0.20,
                    label=r"$\pm 1\sigma$")
    ax.axvline(0, color="r", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel(r"$d_t$ (hidden)"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.bar(t_idx, dose, width=0.9, color="0.35", alpha=0.85)
    ax.set_ylabel(r"MTBS dose $s_k$"); ax.set_xlabel("Months since anchor fire")
    ax.grid(alpha=0.3)

    txt = (f"r   = {params['r']:.3f}/mo   τ_H={1/params['r']:.1f} mo\n"
           f"τ_D = {params['tauD']:.2f} mo\n"
           f"κ   = {params['kappa']:.3f}\n"
           f"γ, a = {params.get('gamma', 0.6):.2f}, {params.get('a', 0.8):.2f}\n"
           f"A1={params.get('A1', 0):.3f}  A2={params.get('A2', 0):.3f}\n"
           f"q_h={params.get('q_h', 0):.2e}  q_d={params.get('q_d', 0):.2e}")
    axes[0].text(0.01, 0.05, txt, transform=axes[0].transAxes, fontsize=8.5,
                 family="monospace", bbox=dict(facecolor="white", alpha=0.85, lw=0.3))

    plt.tight_layout()
    plt.savefig(out_path, dpi=140); plt.close()
    print(f"wrote {out_path}")


def main():
    from grazer.inference.io import load_site, forcing_dose_on_grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="rx_site_333")
    ap.add_argument("--fit-type", default="ekf",
                    choices=["ekf", "two", "one", "chosen"])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--c2", type=float, default=0.0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--plots-dir", default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    data_dir  = Path(args.data_dir)  if args.data_dir  else here / "data" / "timeseries"
    plots_dir = Path(args.plots_dir) if args.plots_dir else here / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    t_idx, y_obs, meta = load_site(args.site, data_dir)
    dose = forcing_dose_on_grid(meta, t_idx)

    fit_suffix = {"ekf": "_ekf", "two": "_2state", "one": "_1state", "chosen": ""}[args.fit_type]
    fit_path = data_dir / f"{args.site}_kalman_fit{fit_suffix}.json"
    fit = json.loads(fit_path.read_text())
    print(f"[{args.site}] seeded from {fit_path.name}  "
          f"(r={fit['parameters']['r']:.3f}, τ_D={fit['parameters'].get('tauD', float('inf')):.2f},"
          f" κ={fit['parameters'].get('kappa', 0):.3f})")
    params = dict(fit["parameters"])
    if not np.isfinite(params.get("tauD", np.inf)):
        params["tauD"] = 1e6
    params.setdefault("kappa", 0.0)
    params.setdefault("q_h", 0.0)
    params.setdefault("q_d", 0.0)
    params.setdefault("R",  0.0)

    ensemble = []
    for s in range(args.n_seeds):
        rng = np.random.default_rng(2026 + s)
        ens = simulate_reduced_nonlinear(t_idx.astype(float), dose, params, rng, c2=args.c2)
        ensemble.append(ens)

    out_path = plots_dir / f"reduced_sim_{args.site}.png"
    plot_reduced_simulation(t_idx.astype(float), y_obs, dose, ensemble, params,
                            args.site, out_path)

    y_hat = np.stack([e["y_mean"] for e in ensemble]).mean(axis=0)
    finite = np.isfinite(y_obs)
    err = y_hat[finite] - y_obs[finite]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(y_hat[finite], y_obs[finite])[0, 1])
    print(f"Reduced-sim vs observed NDVI: RMSE={rmse:.4f}  corr={corr:.3f}")


if __name__ == "__main__":
    main()
