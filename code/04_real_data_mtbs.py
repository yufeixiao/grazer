"""
04 — Real MTBS data: per-site 1-state vs EKF fits and detection.

Consumes the JSON reports written by `grazer.inference.fit_sites.fit_one_site`
in `data/mtbs_legacy/fits/`. Produces:

  figures/04_mtbs/<site>_ekf.png            — per-site 4-panel (dose, NDVI+fit,
                                                h, d) for the EKF
  figures/04_mtbs/<site>_1state.png         — per-site 4-panel for the 1-state KF
  figures/04_mtbs_per_site_table.png        — summary table per site
  figures/04_mtbs_cusum_postfire.png        — CUSUM reset at t=0 (anchor fire),
                                                window [-12, +60] months
  figures/04_mtbs_ridge_spread.png          — multi-start τ_D / κ spread

Run:  python notebooks/04_real_data_mtbs.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

_here = pathlib.Path(__file__).resolve()
_repo = None
for cand in [_here, *_here.parents]:
    if (cand / "grazer").is_dir():
        _repo = cand
        sys.path.insert(0, str(cand))
        break

from grazer.data.mtbs_legacy import SITE_IDS, DATA_DIR as MTBS_DIR, load_site  # noqa: E402
from grazer.inference.detection import (                                       # noqa: E402
    edited_innovations, standardized_innovations,
)
from grazer.inference.fit import fit_kalman                                    # noqa: E402
from grazer.inference.init import irls_prefire                                 # noqa: E402
from grazer.inference.io import plot_fit                                       # noqa: E402

FITS_DIR = MTBS_DIR / "fits"
FIG_DIR = _repo / "grazer" / "figures"
FIG_DIR.mkdir(exist_ok=True)
PER_SITE_DIR = FIG_DIR / "04_mtbs"
PER_SITE_DIR.mkdir(exist_ok=True)


def load_reports():
    reports = {}
    for sid in SITE_IDS:
        path = FITS_DIR / f"{sid}_fit.json"
        if not path.exists():
            continue
        with open(path) as f:
            reports[sid] = json.load(f)
    return reports


def refit_for_traces(sid):
    """Re-run 1-state and EKF fits so we have innovations and state histories
    for plotting. Summary JSONs don't store these arrays."""
    t, y, meta = load_site(sid)
    pre = (t < 0) & np.isfinite(y)
    try:
        pf = irls_prefire(t[pre][-36:], y[pre][-36:], min_months=12)
    except ValueError:
        pf = None
    f1 = fit_kalman(t, y, meta=meta, model="one", prefit=pf,
                    gate_outliers=True, gate_alpha=0.01)
    fe = fit_kalman(t, y, meta=meta, model="ekf", prefit=pf,
                    gate_outliers=True, gate_alpha=0.01)
    return t, y, meta, f1, fe, pf


def plot_summary_table(reports):
    rows = []
    for sid in SITE_IDS:
        if sid not in reports:
            continue
        r = reports[sid]
        p = r["fit_ekf_params"]
        cyc = r["cyclicality"]
        pegs = r["floor_pegs"]
        rows.append([
            sid.replace("rx_site_", ""),
            cyc["class"],
            cyc["n_burns"],
            f"{cyc['cv_interval']:.2f}",
            f"{p.get('tauD', float('nan')):.1f}",
            f"{p.get('kappa', float('nan')):.2e}",
            f"{r['dAIC_1state_minus_ekf']:+.1f}",
            f"{r['ljungbox_p_1state']:.3f}",
            f"{r['ljungbox_p_ekf']:.3f}",
            str(r["cusum_latency_mo_ekf"]) if r["cusum_latency_mo_ekf"] is not None else "—",
            "Y" if pegs["kappa_at_floor"] else "n",
        ])

    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 1.2))
    ax.axis("off")
    col = [r"site", r"class", r"$n_{\mathrm{burns}}$", r"CV",
           r"$\tau_D$ (mo)", r"$\kappa$", r"$\Delta$AIC",
           r"$p_{\mathrm{LB}}$ (1-st)", r"$p_{\mathrm{LB}}$ (EKF)",
           r"CUSUM lat (mo)", r"$\kappa$-floor?"]
    tab = ax.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.3)
    ax.set_title(r"MTBS rx sites — Phase 5 fits + detection "
                 r"($\Delta$AIC $=$ AIC$_{\mathrm{1-state}} -$ AIC$_{\mathrm{EKF}}$)",
                 fontsize=11, pad=10)
    out = FIG_DIR / "04_mtbs_per_site_table.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


def _cusum_from_t0(fit, t_anchor_idx: int) -> dict:
    """CUSUM started at t=0 (anchor fire). Accumulating only post-anchor
    innovations answers the Phase 5 question: does the EKF detect the
    anchor burn?"""
    z = standardized_innovations(fit)
    z_edit = edited_innovations(np.nan_to_num(z, nan=0.0))
    n = len(z_edit)
    pos = np.zeros(n); neg = np.zeros(n)
    drift = 0.5
    for k in range(t_anchor_idx, n):
        prev_p = pos[k - 1] if k > t_anchor_idx else 0.0
        prev_n = neg[k - 1] if k > t_anchor_idx else 0.0
        pos[k] = max(0.0, prev_p + z_edit[k] - drift)
        neg[k] = min(0.0, prev_n + z_edit[k] + drift)
    return {"pos": pos, "neg": neg}


def plot_postfire_cusum(site_traces):
    n = len(site_traces)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.3 * rows), sharex=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)
    h = 5.0
    t_lo, t_hi = -12, 60
    for i, (sid, data) in enumerate(site_traces.items()):
        ax = axes[i // cols, i % cols]
        t = data["t"]
        t_anchor = int(np.argmin(np.abs(t - 0.0)))
        c1 = _cusum_from_t0(data["f1"], t_anchor)
        ce = _cusum_from_t0(data["fe"], t_anchor)
        mask = (t >= t_lo) & (t <= t_hi)
        ax.plot(t[mask], c1["pos"][mask], "-", color="C0", lw=1.3,
                label=r"1-state $S^+$")
        ax.plot(t[mask], c1["neg"][mask], "--", color="C0", lw=0.9)
        ax.plot(t[mask], ce["pos"][mask], "-", color="C3", lw=1.3,
                label=r"EKF $S^+$")
        ax.plot(t[mask], ce["neg"][mask], "--", color="C3", lw=0.9)
        ax.axhline(h, color="0.6", lw=0.6); ax.axhline(-h, color="0.6", lw=0.6)
        ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(sid.replace("rx_site_", "site "), fontsize=9, pad=2)
        ax.set_xlim(t_lo, t_hi); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
        if i // cols == rows - 1:
            ax.set_xlabel("months since anchor fire")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(r"Post-anchor CUSUM: reset at $t=0$, window $[-12,+60]$ mo. "
                 r"$\pm 5$ thresholds in grey.",
                 y=1.0, fontsize=12)
    out = FIG_DIR / "04_mtbs_cusum_postfire.png"
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    print(f"wrote {out}")


def plot_ridge_spread(reports):
    sids = [s for s in SITE_IDS if s in reports]
    tauD_p16 = np.array([reports[s]["multistart_spread"]["tauD"]["p16"] for s in sids])
    tauD_p50 = np.array([reports[s]["multistart_spread"]["tauD"]["p50"] for s in sids])
    tauD_p84 = np.array([reports[s]["multistart_spread"]["tauD"]["p84"] for s in sids])
    k_p16 = np.array([reports[s]["multistart_spread"]["kappa"]["p16"] for s in sids])
    k_p50 = np.array([reports[s]["multistart_spread"]["kappa"]["p50"] for s in sids])
    k_p84 = np.array([reports[s]["multistart_spread"]["kappa"]["p84"] for s in sids])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(sids))

    ax = axes[0]
    ax.errorbar(x, tauD_p50, yerr=[tauD_p50 - tauD_p16, tauD_p84 - tauD_p50],
                fmt="o", color="C2", capsize=4)
    ax.set_yscale("log"); ax.set_ylabel(r"$\tau_D$ (mo)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("rx_site_", "") for s in sids], rotation=45)
    ax.set_title(r"Multi-start spread of $\tau_D$ (p16/p50/p84)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(x, k_p50, yerr=[k_p50 - k_p16, k_p84 - k_p50],
                fmt="o", color="C3", capsize=4)
    ax.set_yscale("log"); ax.set_ylabel(r"$\kappa$")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("rx_site_", "") for s in sids], rotation=45)
    ax.set_title(r"Multi-start spread of $\kappa$ (p16/p50/p84)")
    ax.grid(alpha=0.3)

    fig.suptitle(r"EKF identifiability ridge — per-site multi-start range",
                 fontsize=12)
    out = FIG_DIR / "04_mtbs_ridge_spread.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


def write_markdown_summary(reports):
    lines = [
        r"# Phase 5 MTBS summary",
        "",
        f"Generated {datetime.now():%Y-%m-%d}. {len(reports)} sites.",
        "",
        r"| site | class | $n_{\mathrm{burns}}$ | CV | $\tau_D$ | $\kappa$ | $\Delta$AIC | $p_{\mathrm{LB}}$ 1-st | $p_{\mathrm{LB}}$ EKF | CUSUM lat (mo, EKF) | $\kappa$-floor? |",
        r"|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    n_lb1 = n_better = n_cusum_at_fire = 0
    for sid in SITE_IDS:
        if sid not in reports: continue
        r = reports[sid]
        p = r["fit_ekf_params"]; cyc = r["cyclicality"]
        pegs = r["floor_pegs"]
        lat = r["cusum_latency_mo_ekf"]
        lb1 = r["ljungbox_p_1state"]; lbe = r["ljungbox_p_ekf"]
        if lb1 < 0.05: n_lb1 += 1
        if (lb1 < 0.05) and (lbe > lb1): n_better += 1
        if lat is not None and -30 <= lat <= 30: n_cusum_at_fire += 1
        lines.append(
            f"| {sid.replace('rx_site_','')} "
            f"| {cyc['class']} | {cyc['n_burns']} | {cyc['cv_interval']:.2f} "
            f"| {p.get('tauD', float('nan')):.1f} "
            f"| {p.get('kappa', float('nan')):.2e} "
            f"| {r['dAIC_1state_minus_ekf']:+.1f} "
            f"| {lb1:.3f} | {lbe:.3f} "
            f"| {'—' if lat is None else f'{lat:+.0f}'} "
            f"| {'Y' if pegs['kappa_at_floor'] else 'n'} |"
        )
    n = len(reports)
    lines += [
        "",
        r"## Acceptance criteria (spec \S9 Phase 5)",
        "",
        rf"- 1-state Ljung-Box $p < 0.05$ at $N/M$ sites: $N = {n_lb1}$ / $M = {n}$",
        rf"- EKF $p_{{\mathrm{{LB}}}}$ strictly better than 1-state (and 1-state rejected) at: {n_better} / {n}",
        rf"- Full-record CUSUM crossing within $\pm 30$ months of anchor fire (EKF): {n_cusum_at_fire} / {n}",
        "",
        "Full-record CUSUM latency is negative at most sites because these are "
        "repeatedly-burned rx compartments: the first crossing catches an "
        "earlier fire, not the anchor fire. The post-fire CUSUM figure resets "
        "accumulation at $t=0$ — that is the correct detection diagnostic for "
        "the anchor event.",
        "",
        r"## Ridge / identifiability",
        "",
    ]
    for sid in SITE_IDS:
        if sid not in reports: continue
        sp = reports[sid]["multistart_spread"]
        lines.append(rf"- {sid}: $\tau_D$ geom-ratio $= {sp['tauD']['geom_ratio']:.2f}$, "
                     rf"$\kappa$ geom-ratio $= {sp['kappa']['geom_ratio']:.2f}$")
    out = FITS_DIR / "phase5_summary.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def plot_per_site_fits(site_traces):
    """One 4-panel PNG per site per model, via inference.io.plot_fit.
    Panels: NDVI + fit, h(t), d(t), dose s_k."""
    for sid, d in site_traces.items():
        out1 = PER_SITE_DIR / f"{sid}_1state.png"
        oute = PER_SITE_DIR / f"{sid}_ekf.png"
        # io.plot_fit expects `fit["y"]` and `fit["t_idx"]`, which fit_kalman sets.
        plot_fit(d["f1"]["t_idx"], d["f1"]["y"], d["f1"], d["meta"], out1)
        plot_fit(d["fe"]["t_idx"], d["fe"]["y"], d["fe"], d["meta"], oute)


def main():
    reports = load_reports()
    if not reports:
        raise SystemExit("no fit JSONs found; run `python -m grazer.inference.fit_sites` first")
    print(f"loaded {len(reports)} site reports")

    print("refitting for innovation + state traces...")
    site_traces = {}
    for sid in SITE_IDS:
        if sid not in reports: continue
        t, y, meta, f1, fe, pf = refit_for_traces(sid)
        site_traces[sid] = {"t": t, "y": y, "meta": meta, "f1": f1, "fe": fe, "prefit": pf}
        print(f"  {sid} ok")

    plot_per_site_fits(site_traces)
    plot_summary_table(reports)
    plot_postfire_cusum(site_traces)
    plot_ridge_spread(reports)
    write_markdown_summary(reports)


if __name__ == "__main__":
    main()
