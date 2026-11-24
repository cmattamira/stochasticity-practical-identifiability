"""Figure 6: simulated trajectories under four data-generating mechanisms.

One panel each, with the deterministic solution drawn over the top:

    CTMC          the saved stochastic trajectories
    Gaussian      I + I*eps, eps ~ N(0, sigma), independent at each time
    empirical     I * (1 + eps), eps drawn from the residual pool for that
                  phase and prevalence (empirical_noise.py)
    warped        a * I_ODE(t - Delta_t) (time_shift.py, peak_scaling.py,
                  warp.py)

Two of the four panels needed regenerating. The empirical panel was produced
with a 100-bin residual lookup that could not be filled from the available
prevalence values, so most draws found an empty pool and added no noise at all
-- the panel showed something much closer to a noiseless ODE than to an
empirical noise model. The warped panel used one of several warping
implementations in the code, and not the one the revision describes. The CTMC
and Gaussian panels are unchanged in method and are regenerated here only so
that all four come from one script.

    python fig6_regen.py --data <dir> --summary summary_N1000.csv --out .
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import empirical_noise as en
from time_shift import draw_time_shift
from peak_scaling import load_scaling_moments, draw_scaling_factor
from warp import ode_prevalence_extended, warp_exact

N0 = 1000
I0 = 5
T_OBS = 100
init_cond = np.array([N0 - I0, I0, 0, 0])
t_eval = np.linspace(1.0, float(T_OBS), T_OBS)


def dydt(t, y, par):
    par = np.abs(par)
    alpha, beta = par[0], par[1]
    S, I, R, C = y
    return np.array([-beta*S*I, beta*S*I - alpha*I, alpha*I, beta*S*I])


def prevalence_of(par, times):
    soln = solve_ivp(dydt, (times[0], times[-1]), init_cond, t_eval=times,
                     args=(par,))
    return soln.y[1]


# ------------------------------------------------------------ the four panels
def ctmc_curves(ctmc_dir, alpha, beta, n_curves):
    """Prevalence series from the saved CTMC trajectories, one row each."""
    name = (f"alpha{str(alpha).replace('.', 'p')}"
            f"_beta{str(beta).replace('.', 'p')}.csv")
    df = pd.read_csv(os.path.join(ctmc_dir, name))
    df = df[df["time"].isin(t_eval)]
    wide = df.pivot_table(index="trajectory", columns="time", values="I")
    return wide.to_numpy()[:n_curves]


def gaussian_curves(I_ode, n_curves, sigma):
    return np.array([I_ode + I_ode * np.random.normal(0, sigma, size=len(I_ode))
                     for _ in range(n_curves)])


def empirical_curves(lookup, alpha, beta, I_ode, n_curves):
    """I * (1 + eps), matching how case2.py builds a synthetic dataset."""
    pools = en.pool_index(lookup, alpha, beta)
    t_peak = t_eval[np.argmax(I_ode)]
    out = np.empty((n_curves, len(I_ode)))
    for k in range(n_curves):
        for i, t in enumerate(t_eval):
            phase = "Before Peak" if t < t_peak else "After Peak"
            eps = en.draw_residual(lookup, pools, phase, I_ode[i])
            out[k, i] = I_ode[i] * (1 + eps)
    return out


def warped_curves(alpha, beta, moments_by_R0, n_curves):
    delta_t = draw_time_shift(alpha, beta, N0, I0, size=n_curves)
    a = draw_scaling_factor(alpha, beta, N0, size=n_curves,
                            moments_by_R0=moments_by_R0)
    pad = float(np.abs(delta_t).max()) + 1.0
    prevalence = ode_prevalence_extended(alpha, beta, init_cond, t_eval, pad)
    return np.array([warp_exact(prevalence, t_eval, delta_t[k], a[k])
                     for k in range(n_curves)])


# -------------------------------------------------------------------- figure
def make_figure(panels, I_ode, path, share_y=True):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True,
                             sharey=share_y)
    for ax, (name, curves) in zip(axes.ravel(), panels):
        for row in curves:
            ax.plot(t_eval, row, lw=0.6, alpha=0.25, color="#3b6ea5")
        ax.plot(t_eval, I_ode, "k--", lw=1.8, label="deterministic ODE")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("time (days)")
        ax.set_ylabel("infected individuals")
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.0004)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--curves", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--params", required=True, help="params_combos.csv")
    p.add_argument("--ctmc", required=True, help="directory of CTMC trajectory CSVs")
    p.add_argument("--summary", required=True, help="summary_N1000.csv")
    p.add_argument("--out", default=".")
    p.add_argument("--separate-y", action="store_true",
                   help="give each panel its own y scale")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    alpha, beta, n = args.alpha, args.beta, args.curves

    I_ode = prevalence_of((alpha, beta), t_eval)
    moments = load_scaling_moments(args.summary)

    print("building the residual lookup (this is the slow part) ...")
    residuals = en.build_residuals(pd.read_csv(args.params), args.ctmc,
                                   prevalence_of, t_eval, N0)
    lookup = en.build_lookup(residuals)
    print(f"  {len(residuals)} residuals, {len(lookup['pools'])} pools")

    panels = [
        ("CTMC", ctmc_curves(args.ctmc, alpha, beta, n)),
        (f"Gaussian noise ($\\sigma = {args.sigma}$)",
         gaussian_curves(I_ode, n, args.sigma)),
        ("Empirical residual resampling",
         empirical_curves(lookup, alpha, beta, I_ode, n)),
        ("Warped ODE", warped_curves(alpha, beta, moments, n)),
    ]

    for name, curves in panels:
        sd_at_peak = curves[:, int(np.argmax(I_ode))].std(ddof=1)
        print(f"  {name:38s} n={len(curves):4d}  "
              f"sd at the ODE peak: {sd_at_peak:7.2f}")

    make_figure(panels, I_ode, os.path.join(args.out, "Fig6_regen"),
                share_y=not args.separate_y)
    print(f"\nwritten to {args.out}: Fig6_regen.pdf/.png")
