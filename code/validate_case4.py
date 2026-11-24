"""Validation for the warped-ODE generator, against section 6 of the spec.

This is not the coverage loop. It generates M datasets at ONE parameter pair,
fits each, and reports the coefficient of variation of the recovered
parameters -- which tests the generator and the fitter without the nested loop
on top.

The targets in the spec, at alpha = 0.10, beta = 0.0004 (R0 = 4.00), M = 300:

    CTMC (saved trajectories)        CV(alpha) 5.05   CV(beta)  9.65
    warp, empirical Delta_t and a              5.98            11.57
    warp, theory Delta_t, empirical a          6.13             9.85
    warp, theory Delta_t, a = 1                3.93             9.77

Two reasons the warp rows may not reproduce exactly. The targets were computed
with linear interpolation of the shifted curve; this generator re-solves the
ODE at t - Delta_t instead, which differs on the rising limb where the old way
had to extrapolate backwards along an exponential. And the published shifts
were extracted from peak times on a one-day grid, so the empirical rows inherit
that quantisation while the theory rows do not.

    python validate_case4.py
"""

import math
import random

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from time_shift import draw_time_shift, shift_moments
from warp import ode_prevalence_extended, warp_exact

N0, I0 = 1000, 5
T_OBS = 100
MAJOR_FRACTION = 0.25
init_cond = np.array([N0 - I0, I0, 0, 0])
t_eval = np.linspace(1.0, float(T_OBS), T_OBS)
tdata = np.arange(0, T_OBS)


def dydt(t, y, par):
    par = np.abs(par)
    alpha, beta = par[0], par[1]
    S, I, R, C = y
    return np.array([-beta*S*I, beta*S*I - alpha*I, alpha*I, beta*S*I])


def objective_fun(par, tdata, ydatap, init_cond):
    """Copied from case3.py so the fitting is identical to the other arms."""
    soln = solve_ivp(dydt, (tdata[0], tdata[-1]), init_cond,
                     t_eval=tdata, args=(par,))
    return np.sum((soln.y[1] - ydatap) ** 2)


def simulate_ctmc(alpha, beta, rng):
    """Daily prevalence and final size. Same simulator as fig7_regen.py."""
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


def deterministic_final_size(alpha, beta):
    from scipy.optimize import brentq
    R0 = beta * N0 / alpha
    S0 = N0 - I0
    f = lambda S: S - S0 * np.exp(-R0 * (N0 - S) / N0)
    return S0 - brentq(f, 1e-12, S0 - 1e-12, xtol=1e-12, rtol=1e-14)


def fit_all(datasets, start):
    out = np.zeros((2, len(datasets)))
    for m, y in enumerate(datasets):
        res = minimize(objective_fun, start, args=(tdata, y, init_cond),
                       method="Nelder-Mead")
        out[:, m] = np.abs(res.x)
    return out


def cv(fits):
    return 100 * fits.std(axis=1, ddof=1) / fits.mean(axis=1)


def warped(alpha, beta, delta_t, a):
    """M warped series, the ODE solved once and read at t - Delta_t."""
    pad = float(np.abs(delta_t).max()) + 1.0
    prevalence = ode_prevalence_extended(alpha, beta, init_cond, t_eval, pad)
    return [warp_exact(prevalence, t_eval, delta_t[m], a[m])
            for m in range(len(delta_t))]


if __name__ == "__main__":
    import sys
    alpha, beta = 0.10, 0.0004
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    # Per-trajectory samples written by fig7_regen.py --save-samples,
    # relative to this file.
    import os
    samples = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                           "fig7", "samples", "samples_N1000_alpha0p1_beta0p0004.csv")

    np.random.seed(20261117)
    rng = random.Random(20261117)

    print(f"alpha = {alpha}, beta = {beta}, R0 = {beta*N0/alpha:.2f}, M = {M}\n")

    mean_dt, sd_dt = shift_moments(alpha, beta, N0, I0)
    print(f"theoretical shift:  mean {mean_dt:+.3f} d   sd {sd_dt:.3f} d")
    print(f"  (the spec's section 6 says +0.92 and 2.40; those numbers are wrong)\n")

    d = pd.read_csv(samples)
    major = d["major"].to_numpy()
    emp_dt = d.loc[major, "dt_grid"].to_numpy()
    emp_a = d.loc[major, "a_grid"].to_numpy()
    print(f"empirical, major outbreaks only (n = {major.sum()}):")
    print(f"  shift mean {emp_dt.mean():+.3f} d  sd {emp_dt.std(ddof=1):.3f} d")
    print(f"  scale mean {emp_a.mean():.4f}  sd {emp_a.std(ddof=1):.4f}\n")

    rows = []

    # 1. CTMC, major outbreaks only
    cutoff = MAJOR_FRACTION * deterministic_final_size(alpha, beta)
    ctmc = []
    while len(ctmc) < M:
        daily, C_end = simulate_ctmc(alpha, beta, rng)
        if C_end >= cutoff:
            ctmc.append(daily)
    rows.append(("CTMC, major outbreaks", cv(fit_all(ctmc, [alpha, beta])), "5.05 / 9.65"))

    # 2. warp, both empirical
    dt = np.random.choice(emp_dt, size=M)
    a = np.random.choice(emp_a, size=M)
    rows.append(("warp, empirical dt and a",
                 cv(fit_all(warped(alpha, beta, dt, a), [alpha, beta])), "5.98 / 11.57"))

    # 3. warp, theory shift, empirical scale
    dt = draw_time_shift(alpha, beta, N0, I0, size=M)
    a = np.random.choice(emp_a, size=M)
    rows.append(("warp, theory dt, empirical a",
                 cv(fit_all(warped(alpha, beta, dt, a), [alpha, beta])), "6.13 / 9.85"))

    # 4. warp, theory shift, no scaling
    dt = draw_time_shift(alpha, beta, N0, I0, size=M)
    a = np.ones(M)
    rows.append(("warp, theory dt, a = 1",
                 cv(fit_all(warped(alpha, beta, dt, a), [alpha, beta])), "3.93 / 9.77"))

    print(f"{'condition':32s} {'CV(alpha)':>10s} {'CV(beta)':>9s}   {'spec target':>14s}")
    for name, c, target in rows:
        print(f"{name:32s} {c[0]:10.2f} {c[1]:9.2f}   {target:>14s}")
    print(f"\nMonte Carlo error at M = {M} is about 4% relative.")
