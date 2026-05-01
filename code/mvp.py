from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from sim.reduced import simulate_reduced_nonlinear
from inference.deseason import deseasonalize
from inference.fit_oem import fit_oem
from inference.ndvi_metrics import ljungbox_residual, ndvi_rmse, ndvi_nse
from inference.state_estimation import ekf_fit_eooth, cusum_trace
from plot_style import apply_style, figsize_2col
apply_style()

FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)
print('root:', ROOT)

T = 144
fire_k = 24
t_idx = np.arange(T) - fire_k
dose = np.zeros(T); dose[fire_k] = 3.0

params_true = dict(
    r=0.30, tauD=100.0, kappa=0.20,
    gamma=0.6, a=0.22,   # rescaled from 0.8 to match the framework eq (7)
    # forward-Euler discretization d_jump = a*s*dt; preserves slow-channel SNR
    h0=0.70, c0=0.0,
    A1=0.04, A2=0.02,
    q_h=1e-4, q_d=1e-4, R=1e-3,
)

rng = np.random.default_rng(7)
sim = simulate_reduced_nonlinear(t_idx.astype(float), dose, params_true, rng)
y = sim['y']
print('post-fire visible drop  h(0)=%.3f  h(+1)=%.3f  h(+12)=%.3f  h(+60)=%.3f' %
      (sim['h'][fire_k], sim['h'][fire_k+1], sim['h'][fire_k+12], sim['h'][fire_k+60]))
print('post-fire hidden build  d(0)=%.3f  d(+1)=%.3f  d(+12)=%.3f  d(+60)=%.3f' %
      (sim['d'][fire_k], sim['d'][fire_k+1], sim['d'][fire_k+12], sim['d'][fire_k+60]))

fig, axes = plt.subplots(4, 1, figsize=(figsize_2col[0], 7.0), sharex=True,
                          gridspec_kw={'height_ratios': [1.3, 1, 1, 0.5]})
ax = axes[0]
ax.plot(t_idx, y, 'o', ms=3, color='0.25', label=r'synthetic $y_k$')
ax.plot(t_idx, sim['y_mean'], '-', color='C1', lw=1.4, label=r'noiseless mean $\mathbb{E}[y_k]$')
ax.axvline(0, color='r', ls='--', lw=0.8); ax.set_ylabel('NDVI'); ax.legend(loc='lower right')
ax.set_title(r'D1 synthetic record: $\kappa=%.2f,\ \tau_D=%.0f,\ r=%.2f$' %
             (params_true['kappa'], params_true['tauD'], params_true['r']))

ax = axes[1]
ax.plot(t_idx, sim['h'], '-', color='C2'); ax.axhline(params_true['h0'], color='C2', ls=':', lw=0.8)
ax.axvline(0, color='r', ls='--', lw=0.8); ax.set_ylabel(r'$h_k$ (visible)')

ax = axes[2]
ax.plot(t_idx, sim['d'], '-', color='C3'); ax.axhline(0, color='C3', ls=':', lw=0.8)
ax.axvline(0, color='r', ls='--', lw=0.8); ax.set_ylabel(r'$d_k$ (hidden)')

ax = axes[3]
ax.bar(t_idx, dose, width=0.9, color='0.35'); ax.set_ylabel(r'$s_k$'); ax.set_xlabel('months since impulse')

plt.tight_layout()
plt.savefig(FIG_DIR / 'mvp_synthetic_trajectory.png', dpi=140)
plt.show()

anomaly, season_params = deseasonalize(t_idx, y)

print(f"Fitted seasonal params: c0={season_params['c0']:.4f}, "
      f"A1={season_params['A1']:.4f}, A2={season_params['A2']:.4f}")
print(f"True seasonal params:   c0={params_true['c0']:.4f}, "
      f"A1={params_true['A1']:.4f}, A2={params_true['A2']:.4f}")
print(f"Pre-fire anomaly mean: {anomaly[t_idx < 0].mean():.4f} (should be ~0)")

fig, axes = plt.subplots(2, 1, figsize=(figsize_2col[0], 4.0), sharex=True)
ax = axes[0]
ax.plot(t_idx, y, 'o', ms=3, color='0.25', label=r'$y_k$ (raw)')
season_fit = season_params['c0'] + season_params['A1'] * np.sin(2*np.pi*t_idx/12) \
           + season_params['A2'] * np.cos(2*np.pi*t_idx/12)
ax.plot(t_idx, season_fit, '-', color='C1', lw=1.4, label=r'$\hat y_k^{\mathrm{season}}$')
ax.axvline(0, color='r', ls='--', lw=0.8); ax.set_ylabel('NDVI'); ax.legend(loc='lower right')
ax.set_title('Stage 0: seasonal detrending')

ax = axes[1]
ax.plot(t_idx, anomaly, 'o', ms=3, color='C0', label=r'$\tilde a_k$ (anomaly)')
ax.plot(t_idx, sim['h'] - params_true['h0'], '-', color='C2', lw=1.2,
        alpha=0.7, label=r'true $h_k - h_0$ (disturbance shape)')
ax.axhline(0, color='0.4', ls=':', lw=0.8)
ax.axvline(0, color='r', ls='--', lw=0.8); ax.set_ylabel('anomaly'); ax.legend(loc='lower right')
ax.set_xlabel('months since impulse')

plt.tight_layout()
plt.savefig(FIG_DIR / 'mvp_synthetic_stage0.png', dpi=140)
plt.show()

from inference.fit import fit_kalman

fit_1 = fit_kalman(t_idx, y, dose=dose, model='one')
fit_e = fit_kalman(t_idx, y, dose=dose, model='ekf')

assert fit_1['converged'] and fit_e['converged'], 'optimizer did not converge'

def _show(label, fit):
    p = fit['params']
    nll = fit['nll']; aic = fit['aic']; k = fit['k_params']
    r = p['r']; tauD = p.get('tauD', float('inf')); kappa = p.get('kappa', 0.0)
    R = p['R']; q_h = p['q_h']; q_d = p.get('q_d', 0.0)
    print(f'{label}: nll={nll:.3f}  AIC={aic:.3f}  k={k}  r={r:.3f}  tauD={tauD:.1f}  '
          f'kappa={kappa:.3f}  R={R:.2e}  q_h={q_h:.2e}  q_d={q_d:.2e}')

_show('1-state', fit_1)
_show('EKF    ', fit_e)

from scipy.stats import chi2

def ljung_box(nu, lags=10):
    nu = np.asarray(nu)
    nu = nu[np.isfinite(nu)]
    nu = nu - nu.mean()
    n = len(nu)
    c0 = float((nu**2).sum())
    Q = 0.0
    for k in range(1, lags + 1):
        ck = float((nu[k:] * nu[:-k]).sum())
        rk = ck / c0
        Q += rk * rk / (n - k)
    Q *= n * (n + 2)
    p = float(1.0 - chi2.cdf(Q, df=lags))
    return float(Q), p

LAGS = 10
nu_1 = fit_1['extras']['nu']
nu_e = fit_e['extras']['nu']

Q1, p1 = ljung_box(nu_1, lags=LAGS)
Qe, pe = ljung_box(nu_e, lags=LAGS)
dAIC = fit_1['aic'] - fit_e['aic']

print(f'Delta AIC (1-state minus EKF) = {dAIC:.2f}   [pass if > 10]')
print(f'Ljung-Box(10) on 1-state innovations: Q = {Q1:.2f}, p = {p1:.4g}   [pass if < 0.05]')
print(f'Ljung-Box(10) on EKF     innovations: Q = {Qe:.2f}, p = {pe:.4g}   [should be > 0.05]')

acc_dAIC  = dAIC > 10
acc_lb1   = p1 < 0.05
print()
print(f'Acceptance: dAIC > 10  -> {acc_dAIC}')
print(f'Acceptance: 1-state Ljung-Box p < 0.05  -> {acc_lb1}')
assert acc_dAIC and acc_lb1, 'D1 acceptance failed; revisit simulator parameters'

def acf(nu, max_lag=30):
    nu = np.asarray(nu); nu = nu[np.isfinite(nu)]
    nu = nu - nu.mean()
    n = len(nu); c0 = float((nu**2).sum())
    return np.array([float((nu[k:]*nu[:-k]).sum())/c0 for k in range(1, max_lag+1)])

MAX_LAG = 30
acf_1 = acf(nu_1, MAX_LAG); acf_e = acf(nu_e, MAX_LAG)
n_1 = int(np.isfinite(nu_1).sum()); n_e = int(np.isfinite(nu_e).sum())
band_1 = 1.96 / np.sqrt(n_1); band_e = 1.96 / np.sqrt(n_e)

lags = np.arange(1, MAX_LAG+1)
fig, axes = plt.subplots(1, 2, figsize=(figsize_2col[0], 3.0), sharey=True)
for ax, a, band, n_, ttl in [
    (axes[0], acf_1, band_1, n_1, '1-state KF innovations'),
    (axes[1], acf_e, band_e, n_e, '2-state EKF innovations'),
]:
    ax.vlines(lags, 0, a, color='C0', lw=1.4)
    ax.scatter(lags, a, s=10, color='C0')
    ax.axhline(0, color='0.4', lw=0.6)
    ax.axhline( band, color='r', ls='--', lw=0.7, label=r'$\pm 1.96/\sqrt{n}$')
    ax.axhline(-band, color='r', ls='--', lw=0.7)
    ax.set_xlabel('lag'); ax.set_title(ttl); ax.legend(loc='upper right')
axes[0].set_ylabel(r'$\hat\rho_k$')
plt.tight_layout()
plt.savefig(FIG_DIR / 'mvp_synthetic_innovation_acf.png', dpi=140)
plt.show()

yhat_1 = fit_1['extras']['y_pred']
yhat_e = fit_e['extras']['y_pred']
h_1 = np.asarray(fit_1['x_hist'])[:, 0]
xe = np.asarray(fit_e['x_hist'])
h_e = xe[:, 0]; d_e = xe[:, 1]

fig, axes = plt.subplots(3, 1, figsize=(figsize_2col[0], 6.0), sharex=True,
                         gridspec_kw={'height_ratios': [1.3, 1, 1]})

ax = axes[0]
ax.plot(t_idx, y, 'o', ms=3, color='0.25', label=r'synthetic $y_k$')
ax.plot(t_idx, yhat_1, '-', color='C0', lw=1.2, label=r'1-state $\hat y_{k|k-1}$')
ax.plot(t_idx, yhat_e, '-', color='C3', lw=1.2, label=r'EKF $\hat y_{k|k-1}$')
ax.axvline(0, color='r', ls='--', lw=0.8)
ax.set_ylabel('NDVI'); ax.legend(loc='lower right')
ax.set_title('Filter overlays on synthetic record')

ax = axes[1]
ax.plot(t_idx, sim['h'], '-', color='C2', lw=1.4, label=r'true $h_k$')
ax.plot(t_idx, h_1, '--', color='C0', lw=1.1, label=r'1-state $\hat h_{k|k}$')
ax.plot(t_idx, h_e, '--', color='C3', lw=1.1, label=r'EKF $\hat h_{k|k}$')
ax.axvline(0, color='r', ls='--', lw=0.8)
ax.set_ylabel(r'$h_k$'); ax.legend(loc='lower right')

ax = axes[2]
ax.plot(t_idx, sim['d'], '-', color='C3', lw=1.4, label=r'true $d_k$')
ax.plot(t_idx, d_e, '--', color='k', lw=1.1, label=r'EKF $\hat d_{k|k}$')
ax.axvline(0, color='r', ls='--', lw=0.8)
ax.set_ylabel(r'$d_k$'); ax.set_xlabel('months since impulse')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIG_DIR / 'mvp_synthetic_filter_overlays.png', dpi=140)
plt.show()

cu = cusum_trace(xe)

fig, axes = plt.subplots(2, 1, figsize=(figsize_2col[0], 4.0), sharex=True)
ax = axes[0]
ax.plot(t_idx, fit_e['z'], '-', color='0.3', lw=0.9)
ax.scatter(t_idx, fit_e['z'], s=8, color='0.3')
ax.axhline( 3.0, color='r', ls='--', lw=0.8, label=r'$|z|=3$')
ax.axhline(-3.0, color='r', ls='--', lw=0.8)
ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.5)
ax.set_ylabel(r'$z_k$'); ax.legend(loc='upper right'); ax.set_title('Stage 3 detection diagnostics')

ax = axes[1]
ax.plot(t_idx, cu['g_pos'], color='C0', lw=1.2, label=r'$g^+$')
ax.plot(t_idx, cu['g_neg'], color='C3', lw=1.2, label=r'$g^-$')
ax.axhline( 5.0, color='0.4', ls='--', lw=0.8)
ax.axhline(-5.0, color='0.4', ls='--', lw=0.8)
ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.5)
ax.set_ylabel('CUSUM'); ax.set_xlabel('months since impulse'); ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIG_DIR / 'mvp_synthetic_detection.png', dpi=140)
plt.show()

n_z3 = int(np.sum(np.abs(fit_e['z']) > 3.0))
n_alarm = len(cu['alarm_pos']) + len(cu['alarm_neg'])
print(f'|z|>3 at {n_z3} months out of {sum(np.isfinite(fit_e["z"]))} observed')
print(f'CUSUM alarms (|g| > 5): {n_alarm}')

