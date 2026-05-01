"""
Phase 5.2 — per-site real-data fitting pipeline.

For each MTBS rx-burn site:
  1. Load NDVI timeseries + metadata.
  2. IRLS pre-fire fit to pin h_0, A_1, A_2, R_hat (§4.1, Puhm 2020 §2.2.2).
  3. Fit 1-state KF and 2-state EKF with `prefit=` threaded and outlier gating
     at the fire event.
  4. Multi-start EKF (n_starts random starts) to probe the τ_D / κ ridge.
  5. Detection pipeline (χ² gate, edited-innovation CUSUM, Ljung-Box, dAIC).
  6. Persist fit + detection report to data/mtbs_legacy/fits/<site>_fit.json.

Cyclicality class follows spec §5.2: inter-fire interval statistics map each
site to {single-impulse, quasi-periodic, bursty}. Duty cycle is the fraction
of months with a dose event in the post-pre window.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .detection import detection_report
from .fit import fit_kalman, KAPPA_MIN, KAPPA_MAX, Q_D_CEIL_MULT
from .init import irls_prefire
from .io import forcing_dose_on_grid


def cyclicality_metrics(meta: dict) -> dict:
    """Classify a site by its fire recurrence.

    Returns n_burns, mean interval (months), CV, duty cycle, and a class in
    {'single-impulse', 'quasi-periodic', 'bursty'}.
    """
    fr = meta.get("fire_recurrence", {}) or {}
    events = fr.get("events", []) or []
    dates = sorted(datetime.strptime(e["date"], "%Y-%m-%d") for e in events)
    n_burns = len(dates)
    if n_burns <= 1:
        return {
            "n_burns": n_burns,
            "mean_interval_mo": float("nan"),
            "cv_interval": float("nan"),
            "duty_cycle": float("nan"),
            "class": "single-impulse",
        }
    intervals = np.array([(dates[i + 1] - dates[i]).days / 30.44
                          for i in range(n_burns - 1)], dtype=float)
    mean_int = float(intervals.mean())
    cv = float(intervals.std() / max(mean_int, 1e-6))
    span_mo = (dates[-1] - dates[0]).days / 30.44
    duty = float(n_burns / max(span_mo, 1.0))
    if cv < 0.3:
        klass = "quasi-periodic"
    else:
        klass = "bursty"
    return {
        "n_burns":          n_burns,
        "mean_interval_mo": mean_int,
        "cv_interval":      cv,
        "duty_cycle":       duty,
        "class":            klass,
    }


def _rand_theta0_ekf(rng: np.random.Generator) -> np.ndarray:
    """Random init in the EKF free-slot order: r, rho, kappa, q_h, q_d, R.
    (Pinning from `prefit` drops h0/A1/A2 from the optimization vector.)"""
    return np.array([
        np.log(rng.uniform(0.1, 2.0)),    # r
        np.log(rng.uniform(3.0, 25.0)),   # rho
        np.log(rng.uniform(1e-4, 0.5)),   # kappa
        np.log(rng.uniform(1e-5, 1e-2)),  # q_h
        np.log(rng.uniform(1e-5, 1e-2)),  # q_d
        np.log(rng.uniform(5e-4, 5e-3)),  # R
    ])


def _rand_theta0_1state(rng: np.random.Generator) -> np.ndarray:
    """Random init in the 1-state free-slot order: r, q_h, R."""
    return np.array([
        np.log(rng.uniform(0.1, 2.0)),
        np.log(rng.uniform(1e-5, 1e-2)),
        np.log(rng.uniform(5e-4, 5e-3)),
    ])


def _best_multistart(t, y, meta, prefit, model, n_starts, seed,
                     gate_outliers, gate_alpha):
    rng = np.random.default_rng(seed)
    best = None
    tauD_all, kappa_all, r_all = [], [], []
    for _ in range(n_starts):
        theta0 = (_rand_theta0_ekf(rng) if model in ("two", "ekf")
                  else _rand_theta0_1state(rng))
        try:
            fit = fit_kalman(t, y, meta=meta, model=model, prefit=prefit,
                             theta0=theta0, gate_outliers=gate_outliers,
                             gate_alpha=gate_alpha)
        except Exception:
            continue
        if best is None or fit["nll"] < best["nll"]:
            best = fit
        p = fit["params"]
        tauD_all.append(p.get("tauD", np.nan))
        kappa_all.append(p.get("kappa", np.nan))
        r_all.append(p.get("r", np.nan))
    if best is None:
        raise RuntimeError(f"multi-start {model} failed on all starts")
    return best, {
        "tauD":  np.asarray(tauD_all, dtype=float),
        "kappa": np.asarray(kappa_all, dtype=float),
        "r":     np.asarray(r_all, dtype=float),
    }


def _cusum_latency(crossings: dict, t_idx: np.ndarray) -> float | None:
    """First-crossing index (either arm) converted to months since anchor fire.
    t_idx=0 is the anchor month; a negative value means detection *before* the
    fire, a non-negative value means detection at or after.
    """
    idxs = [crossings.get("pos_first"), crossings.get("neg_first")]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return None
    first = int(min(idxs))
    return float(t_idx[first])


def fit_one_site(site_id: str, n_starts: int = 10, seed: int = 0,
                 gate_alpha: float = 0.01, pre_window_max: int = 36,
                 fits_dir: Path | None = None) -> dict:
    """Full Phase 5 pipeline on one site. Returns the site-level report dict."""
    from grazer.data.mtbs_legacy import load_site, DATA_DIR as MTBS_DIR
    if fits_dir is None:
        fits_dir = MTBS_DIR / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    t, y, meta = load_site(site_id)
    cyc = cyclicality_metrics(meta)

    # IRLS on a 12–36 month pre-fire window. Drop sites with <24 usable months.
    pre_mask = (t < 0) & np.isfinite(y)
    t_pre_all = t[pre_mask]; y_pre_all = y[pre_mask]
    n_pre_all = int(pre_mask.sum())
    used_window = min(pre_window_max, n_pre_all)
    if used_window >= 12:
        t_pre = t_pre_all[-used_window:]
        y_pre = y_pre_all[-used_window:]
        try:
            prefit = irls_prefire(t_pre, y_pre, min_months=12)
        except ValueError:
            prefit = None
    else:
        prefit = None
    short_pre = (n_pre_all < 24)

    # At the anchor fire, outlier-gate the EKF measurement update so the
    # fire-month jump is not absorbed as baseline drift (Puhm §2.2.4).
    fit1, ms1 = _best_multistart(t, y, meta, prefit=prefit, model="one",
                                 n_starts=n_starts, seed=seed,
                                 gate_outliers=True, gate_alpha=gate_alpha)
    fit2, ms2 = _best_multistart(t, y, meta, prefit=prefit, model="ekf",
                                 n_starts=n_starts, seed=seed + 1,
                                 gate_outliers=True, gate_alpha=gate_alpha)

    rep1 = detection_report(fit1, alpha=gate_alpha, fit_1state=fit1)
    rep2 = detection_report(fit2, alpha=gate_alpha, fit_1state=fit1)

    # Floor-peg diagnostics — these drove the §5.1 mitigations.
    p = fit2["params"]
    R_floor_eff = max(1e-5, (prefit or {}).get("R_hat", 1e-5))
    pegs = {
        "R_at_Rhat":    bool(prefit is not None
                             and abs(p["R"] - prefit["R_hat"]) < 1e-9),
        "kappa_at_floor": bool(abs(p["kappa"] - KAPPA_MIN) / KAPPA_MIN < 1e-3),
        "kappa_at_ceil":  bool(abs(p["kappa"] - KAPPA_MAX) / KAPPA_MAX < 1e-3),
        "qd_at_ceil":     bool(abs(p["q_d"] - Q_D_CEIL_MULT * R_floor_eff)
                               / (Q_D_CEIL_MULT * R_floor_eff) < 1e-3),
    }

    # Multi-start spread — ridge magnitude.
    def _spread(a: np.ndarray) -> dict:
        finite = np.isfinite(a) & (a > 0)
        if finite.sum() < 2:
            return {"p16": float("nan"), "p50": float("nan"), "p84": float("nan"),
                    "geom_ratio": float("nan"), "n": int(finite.sum())}
        lo, md, hi = np.percentile(np.log(a[finite]), [16, 50, 84])
        return {
            "p16":        float(np.exp(lo)),
            "p50":        float(np.exp(md)),
            "p84":        float(np.exp(hi)),
            "geom_ratio": float(np.exp(hi - lo)),
            "n":          int(finite.sum()),
        }

    spread = {
        "tauD":  _spread(ms2["tauD"]),
        "kappa": _spread(ms2["kappa"]),
        "r":     _spread(ms2["r"]),
    }

    dAIC = float(fit1["aic"] - fit2["aic"])
    cusum_lat_1 = _cusum_latency(rep1["cusum_crossings"], fit1["t_idx"])
    cusum_lat_2 = _cusum_latency(rep2["cusum_crossings"], fit2["t_idx"])

    report = {
        "site_id": site_id,
        "meta": {
            "name": meta.get("name"),
            "state": meta.get("state"),
            "anchor_fire_date": meta.get("anchor_fire", {}).get("date"),
        },
        "cyclicality": cyc,
        "n_obs": int(fit2["n_obs"]),
        "n_pre_obs": int(n_pre_all),
        "pre_window_used": int(used_window),
        "short_pre_window": bool(short_pre),
        "prefit": {k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in (prefit or {}).items()},
        "fit_1state_params": {k: (float(v) if np.isfinite(v) else None)
                              for k, v in fit1["params"].items()},
        "fit_ekf_params":    {k: (float(v) if np.isfinite(v) else None)
                              for k, v in fit2["params"].items()},
        "fit_1state_aic":  float(fit1["aic"]),
        "fit_ekf_aic":     float(fit2["aic"]),
        "fit_1state_nll":  float(fit1["nll"]),
        "fit_ekf_nll":     float(fit2["nll"]),
        "dAIC_1state_minus_ekf": dAIC,
        "ljungbox_p_1state":  float(rep1["ljungbox_p"]),
        "ljungbox_p_ekf":     float(rep2["ljungbox_p"]),
        "cusum_crossings_1state": rep1["cusum_crossings"],
        "cusum_crossings_ekf":    rep2["cusum_crossings"],
        "cusum_latency_mo_1state": cusum_lat_1,
        "cusum_latency_mo_ekf":    cusum_lat_2,
        "n_anomalies_1state":  int(rep1["n_anomalies"]),
        "n_anomalies_ekf":     int(rep2["n_anomalies"]),
        "multistart_spread":   spread,
        "floor_pegs":          pegs,
        "converged_1state":    bool(fit1["converged"]),
        "converged_ekf":       bool(fit2["converged"]),
    }

    out_path = fits_dir / f"{site_id}_fit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def fit_all_sites(site_ids=None, **kwargs) -> list[dict]:
    from grazer.data.mtbs_legacy import SITE_IDS
    site_ids = site_ids or SITE_IDS
    out = []
    for sid in site_ids:
        print(f"=== {sid} ===")
        try:
            rep = fit_one_site(sid, **kwargs)
        except Exception as e:
            print(f"  FAILED: {e}")
            out.append({"site_id": sid, "error": str(e)})
            continue
        p = rep["fit_ekf_params"]
        print(f"  class={rep['cyclicality']['class']:>14s}  "
              f"n_burns={rep['cyclicality']['n_burns']:3d}  "
              f"dAIC={rep['dAIC_1state_minus_ekf']:+.1f}  "
              f"LB_p(1st)={rep['ljungbox_p_1state']:.3f}  "
              f"LB_p(ekf)={rep['ljungbox_p_ekf']:.3f}")
        print(f"  τ_D={p.get('tauD'):.1f}  κ={p.get('kappa'):.2e}  "
              f"CUSUM_lat(ekf)={rep['cusum_latency_mo_ekf']}  "
              f"κ-floor-peg={rep['floor_pegs']['kappa_at_floor']}")
        out.append(rep)
    return out


if __name__ == "__main__":
    fit_all_sites()
