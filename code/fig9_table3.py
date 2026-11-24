"""Figure 9 and Table 3: the spread of best-fit parameters by noise model.

Both come from the same 4000 fits, so they are produced together and cannot
disagree with each other.

For each of four data-generating mechanisms we simulate n_fits synthetic
trajectories at one parameter pair, fit the deterministic model to each, and
keep the recovered (alpha, beta). Figure 9 plots the four clouds; Table 3
reports their coefficients of variation.

    CTMC          direct stochastic simulation, conditioned on a major
                  outbreak, which is the benchmark the others are judged
                  against
    Gaussian      I + I*eps with eps ~ N(0, sigma), the multiplicative noise
                  of the standard Monte Carlo method, at sigma = 0.1 and 0.2
    warped        a * I_ODE(t - Delta_t), with Delta_t from the
                  branching-process limit and a from the measured scaling
                  moments -- see time_shift.py, peak_scaling.py and warp.py

The published versions of this figure and table were made by a different
warped generator: one that added an undocumented autoregressive noise term,
interpolated cubically and resampled the shift and scale from values measured
at the true parameters. See Warp_Generator_Discrepancies.md.

Fitting matches the coverage code exactly -- unweighted sum of squares on
prevalence, Nelder-Mead, started at the true parameters, np.abs on the result
-- so these numbers are comparable with the coverage table.

    python fig9_table3.py --out .
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

from time_shift import draw_time_shift
from peak_scaling import load_scaling_moments, draw_scaling_factor
from warp import ode_prevalence_extended, warp_exact

# ------------------------------------------------------------------ settings
N0 = 1000
I0 = 5
T_OBS = 100
MAJOR_FRACTION = 0.25          # of the deterministic final size
CONF = 0.95                    # for the ellipses drawn on the figure

init_cond = np.array([N0 - I0, I0, 0, 0])
t_eval = np.linspace(1.0, float(T_OBS), T_OBS)   # the data are reported here
tdata = np.arange(0, T_OBS)                      # the model is evaluated here


# ------------------------------------------------------------- the ODE model
def dydt(t, y, par):
    par = np.abs(par)
    alpha, beta = par[0], par[1]
    S, I, R, C = y
    return np.array([-beta*S*I, beta*S*I - alpha*I, alpha*I, beta*S*I])


def ode_prevalence_on(par, times):
    soln = solve_ivp(dydt, (times[0], times[-1]), init_cond, t_eval=times,
                     args=(par,))
    return soln.y[1]


def objective_fun(par, tdata, ydata, init_cond):
    """Copied from the coverage scripts so the fitting is identical."""
    return np.sum((ode_prevalence_on(par, tdata) - ydata) ** 2)


# --------------------------------------------------- the four data mechanisms
def gaussian_datasets(alpha, beta, n_fits, sigma):
    """I + I*eps, eps ~ N(0, sigma). Multiplicative, independent at each time."""
    I = ode_prevalence_on([alpha, beta], t_eval)
    return [I + I * np.random.normal(0, sigma, size=len(I)) for _ in range(n_fits)]


def deterministic_final_size(alpha, beta):
    R0 = beta * N0 / alpha
    S0 = N0 - I0
    f = lambda S: S - S0 * np.exp(-R0 * (N0 - S) / N0)
    return S0 - brentq(f, 1e-12, S0 - 1e-12, xtol=1e-12, rtol=1e-14)


def simulate_ctmc(alpha, beta, rng):
    """One Gillespie realisation: daily prevalence, and the final size."""
    S, I, C = N0 - I0, I0, 0
    t, nxt = 1.0, 1
    daily = np.zeros(T_OBS)
    while I > 0 and t < 5000.0:
        rate_inf, rate_rec = beta * S * I, alpha * I
        total = rate_inf + rate_rec
        t = t - math.log(rng.random()) / total
        while nxt <= T_OBS and t > nxt:
            daily[nxt - 1] = I
            nxt += 1
        if rng.random() * total < rate_inf:
            S -= 1
            I += 1
            C += 1
        else:
            I -= 1
    while nxt <= T_OBS:
        daily[nxt - 1] = I
        nxt += 1
    return daily, C


def ctmc_datasets(alpha, beta, n_fits, rng):
    """n_fits realisations, keeping only those that became major outbreaks."""
    cutoff = MAJOR_FRACTION * deterministic_final_size(alpha, beta)
    out = []
    while len(out) < n_fits:
        daily, C_end = simulate_ctmc(alpha, beta, rng)
        if C_end >= cutoff:
            out.append(daily)
    return out


def warped_datasets(alpha, beta, n_fits, moments_by_R0, rho=0.0):
    """a * I_ODE(t - Delta_t), the ODE solved once and read at the shifted times."""
    delta_t = draw_time_shift(alpha, beta, N0, I0, size=n_fits)
    a = draw_scaling_factor(alpha, beta, N0, size=n_fits,
                            moments_by_R0=moments_by_R0, rho=rho,
                            delta_t=delta_t)
    pad = float(np.abs(delta_t).max()) + 1.0
    prevalence = ode_prevalence_extended(alpha, beta, init_cond, t_eval, pad)
    return [warp_exact(prevalence, t_eval, delta_t[m], a[m]) for m in range(n_fits)]


# ---------------------------------------------------------------- fit and sum
def fit_all(datasets, start):
    """Best-fit (alpha, beta) for each dataset, in the coverage convention."""
    fits = np.zeros((2, len(datasets)))
    for m, y in enumerate(datasets):
        res = minimize(objective_fun, start, args=(tdata, y, init_cond),
                       method="Nelder-Mead")
        fits[:, m] = np.abs(res.x)
    return fits


def cv_percent(fits):
    return 100 * fits.std(axis=1, ddof=1) / fits.mean(axis=1)


# -------------------------------------------------------------------- figure
def confidence_ellipse(ax, x, y, level=CONF, **kw):
    """The ellipse holding `level` of a normal fitted to the cloud.

    Drawn from the eigen-decomposition of the sample covariance rather than
    with a patch, so the geometry is visible in the code: the axes are the
    eigenvectors and the half-lengths are sqrt(chi2 quantile * eigenvalue).
    """
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0, 2 * np.pi, 200)
    radius = np.sqrt(chi2.ppf(level, df=2))
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    pts = vecs @ (radius * np.sqrt(vals)[:, None] * circle)
    ax.plot(pts[0] + x.mean(), pts[1] + y.mean(), **kw)


def make_figure(results, true_par, path, shared_axes=True):
    """The four clouds, one panel each.

    shared_axes puts every panel on the same scale. That matters more than it
    sounds: on independent scales the Gaussian clouds fill their panels and
    look as dispersed as the CTMC one, which is the opposite of the point.
    On a common scale the difference in spread is the figure's whole message.
    Pass --separate-axes to see each cloud's own structure instead.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2))
    if shared_axes:
        allx = np.concatenate([f[0] for _, f in results])
        ally = np.concatenate([f[1] for _, f in results])
        padx = 0.05 * (allx.max() - allx.min())
        pady = 0.05 * (ally.max() - ally.min())
        xlim = (allx.min() - padx, allx.max() + padx)
        ylim = (ally.min() - pady, ally.max() + pady)
    for ax, (name, fits) in zip(axes.ravel(), results):
        x, y = fits[0], fits[1]
        ax.scatter(x, y, s=7, alpha=0.30, color="#3b6ea5", edgecolors="none")
        confidence_ellipse(ax, x, y, color="#333333", ls="--", lw=1.2)
        ax.scatter([x.mean()], [y.mean()], marker="o", s=55, color="#e08214",
                   edgecolors="black", linewidths=0.6, zorder=5,
                   label="mean estimate")
        ax.scatter([true_par[0]], [true_par[1]], marker="*", s=190,
                   color="#1a9850", edgecolors="black", linewidths=0.6,
                   zorder=6, label="true value")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r"$\hat\alpha$")
        ax.set_ylabel(r"$\hat\beta$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if shared_axes:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
    axes[0, 0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.0004)
    p.add_argument("--fits", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--summary", required=True,
                   help="summary_N1000.csv, for the scaling moments")
    p.add_argument("--out", default=".")
    p.add_argument("--separate-axes", action="store_true",
                   help="give each panel its own scale instead of a common one")
    p.add_argument("--replot", default=None,
                   help="Fig9_fits.npz from an earlier run: redraw the figure "
                        "from the saved fits instead of repeating them")
    args = p.parse_args()

    # Redrawing from saved fits costs a second; repeating them costs minutes.
    if args.replot:
        with np.load(args.replot, allow_pickle=False) as d:
            saved = [(str(n), d[f"fits_{i}"]) for i, n in enumerate(d["names"])]
        make_figure(saved, [args.alpha, args.beta],
                    os.path.join(args.out, "Fig9_regen"),
                    shared_axes=not args.separate_axes)
        print("redrawn from", args.replot)
        raise SystemExit(0)

    os.makedirs(args.out, exist_ok=True)
    alpha, beta, n = args.alpha, args.beta, args.fits
    true_par = [alpha, beta]
    moments = load_scaling_moments(args.summary)

    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"alpha = {alpha}, beta = {beta}, R0 = {beta*N0/alpha:.2f}, "
          f"{n} fits per mechanism\n")

    mechanisms = [
        ("CTMC", lambda: ctmc_datasets(alpha, beta, n, rng)),
        (r"Gaussian ($\sigma = 0.1$)", lambda: gaussian_datasets(alpha, beta, n, 0.1)),
        (r"Gaussian ($\sigma = 0.2$)", lambda: gaussian_datasets(alpha, beta, n, 0.2)),
        ("Warped ODE", lambda: warped_datasets(alpha, beta, n, moments)),
    ]

    results, rows = [], []
    for name, make in mechanisms:
        fits = fit_all(make(), true_par)
        results.append((name, fits))
        cva, cvb = cv_percent(fits)
        rows.append(dict(Method=name.replace("$", "").replace("\\", ""),
                         CV_alpha=round(cva, 2), CV_beta=round(cvb, 2),
                         mean_alpha=fits[0].mean(), mean_beta=fits[1].mean()))
        print(f"  {name:28s} CV(alpha) {cva:6.2f}%   CV(beta) {cvb:6.2f}%")

    make_figure(results, true_par, os.path.join(args.out, "Fig9_regen"),
                shared_axes=not args.separate_axes)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "Table3_regen.csv"), index=False)
    np.savez(os.path.join(args.out, "Fig9_fits.npz"),
             names=np.array([r[0] for r in results]),
             **{f"fits_{i}": r[1] for i, r in enumerate(results)})

    print(f"\nwritten to {args.out}: Fig9_regen.pdf/.png, Table3_regen.csv")
    print("\npublished Table 3 for comparison:")
    print("  CTMC 4.77 / 9.64   Gaussian(0.1) 1.90 / 1.00   "
          "Gaussian(0.2) 3.86 / 2.01   Hybrid 6.46 / 11.19")
