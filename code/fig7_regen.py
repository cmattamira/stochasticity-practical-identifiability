"""Regenerate the Fig 7 summaries: shift and amplitude moments by R0 and N.

Replaces the pipeline in 'Data Generation for Fig 7.ipynb' plus the summary
step behind summary_N{N}.csv. Three things are done differently, each of which
changes a number:

  1. The event-driven trajectory is kept separate from the daily observations.
     The simulator knows the exact time at which prevalence peaks; reporting on
     a daily grid is a downsampling we impose afterwards. Both are recorded, so
     the published extraction (peak located on whole days) and the exact one can
     be compared directly.

  2. The ODE is evaluated on the same integer days the CTMC reports, and its
     peak time is found exactly from the solver's dense output rather than being
     quantised to a grid. The original used np.linspace(0, 100, 100), a step of
     1.0101, against a CTMC reporting on integer days; that mismatch is a
     constant per cell, so it biases mean(Delta_t) while leaving sd(Delta_t)
     untouched.

  3. Realisations are classified as major outbreaks by comparing cumulative
     incidence against the DETERMINISTIC final size for that parameter pair,
     C_inf = S0 - S_inf, where S_inf solves S_inf = S0 exp(-R0 (N - S_inf)/N).
     Moments are reported both conditioned and unconditioned.

Every cell is seeded, so this is reproducible -- the original pipeline set no
seeds anywhere, which is why the published points cannot be reproduced exactly
by anyone.

Usage:
    python fig7_regen.py --out <dir> [--trajectories 1000] [--save-trajectories]
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

I0 = 5
T_OBS = 100                 # days of observation, reported at t = 1 .. 100
T_CAP = 5000.0              # safety cap on simulated time
EVENT_CAP = 50_000_000      # safety cap on events
MAJOR_FRACTION = 0.25       # of the deterministic final size


# ------------------------------------------------------------------ the model
def _rhs(t, y, alpha, beta):
    S, I, R, C = y
    return [-beta * S * I, beta * S * I - alpha * I, alpha * I, beta * S * I]


def deterministic_final_size(R0, N):
    """Cumulative NEW infections over the whole epidemic, from the ODE.

    dS/dR = -(R0/N) S integrates to S(t) = S0 exp(-R0 R(t)/N), and at the end
    of the epidemic R_inf = N - S_inf, which leaves a transcendental equation
    in S_inf alone. Note this counts new infections only, so it is S0 - S_inf
    and not N - S_inf; the latter would also count the I0 seeds and the CTMC's
    C column does not.
    """
    S0 = N - I0
    f = lambda S: S - S0 * np.exp(-R0 * (N - S) / N)
    S_inf = brentq(f, 1e-12, S0 - 1e-12, xtol=1e-12, rtol=1e-14)
    return S0 - S_inf


def ode_summary(alpha, beta, N):
    """Prevalence on the daily grid, plus the grid peak and the exact peak."""
    t_obs = np.linspace(1.0, float(T_OBS), T_OBS)
    init = [N - I0, I0, 0.0, 0.0]
    sol = solve_ivp(_rhs, (1.0, float(T_OBS)), init, args=(alpha, beta),
                    method="RK45", rtol=1e-10, atol=1e-12, dense_output=True)
    I_grid = sol.sol(t_obs)[1]
    k = int(np.argmax(I_grid))
    t_grid_peak = float(t_obs[k])

    # exact peak: maximise the dense solution in the bracket around the grid max
    lo = float(t_obs[max(k - 1, 0)])
    hi = float(t_obs[min(k + 1, T_OBS - 1)])
    res = minimize_scalar(lambda t: -float(sol.sol(t)[1]),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    return I_grid, t_grid_peak, float(res.x), float(-res.fun), float(I_grid.max())


# -------------------------------------------------------------- the simulator
def simulate(alpha, beta, N, rng):
    """One SIR realisation.

    Returns the daily prevalence series, the exact peak time and height from
    the event stream, and the cumulative incidence both at day T_OBS and at the
    end of the epidemic.
    """
    S, I, C = N - I0, I0, 0
    t = 1.0
    nxt = 1                           # next integer day to record
    daily = np.zeros(T_OBS)

    peak_I = I                        # running maximum over the continuous path
    peak_t = t                        # first attainment of that maximum
    C_at_obs = None
    events = 0

    while I > 0 and t < T_CAP and events < EVENT_CAP:
        rate_inf = beta * S * I
        rate_rec = alpha * I
        total = rate_inf + rate_rec
        t_next = t - math.log(rng.random()) / total

        while nxt <= T_OBS and t_next > nxt:
            daily[nxt - 1] = I
            nxt += 1
            if nxt > T_OBS and C_at_obs is None:
                C_at_obs = C

        t = t_next
        events += 1
        if rng.random() * total < rate_inf:
            S -= 1
            I += 1
            C += 1
            if I > peak_I:            # strict: first attainment of the maximum
                peak_I = I
                peak_t = t
        else:
            I -= 1

    while nxt <= T_OBS:               # epidemic over before the window closed
        daily[nxt - 1] = I
        nxt += 1
    if C_at_obs is None:
        C_at_obs = C

    return daily, peak_t, peak_I, C_at_obs, C


# ------------------------------------------------------------------ the sweep
def moments(dt, a):
    if len(dt) < 2:
        return dict(mean_dt=np.nan, std_dt=np.nan, mean_a=np.nan, std_a=np.nan)
    return dict(mean_dt=float(np.mean(dt)), std_dt=float(np.std(dt, ddof=1)),
                mean_a=float(np.mean(a)), std_a=float(np.std(a, ddof=1)))


def run_cell(alpha, beta_base, N, n_traj, seed, save_dir=None):
    beta = beta_base * (1000.0 / N)          # hold R0 fixed as N varies
    R0 = beta * N / alpha
    C_inf = deterministic_final_size(R0, N)
    I_grid, tp_grid, tp_exact, peak_exact, peak_grid = ode_summary(alpha, beta, N)

    rng = random.Random(seed)
    t_obs = np.linspace(1.0, float(T_OBS), T_OBS)

    dt_grid, dt_exact, a_grid, a_exact, C_obs, C_end = [], [], [], [], [], []
    for _ in range(n_traj):
        daily, pt, pI, c_obs, c_end = simulate(alpha, beta, N, rng)
        dt_grid.append(float(t_obs[int(np.argmax(daily))]) - tp_grid)
        dt_exact.append(pt - tp_exact)
        a_grid.append(float(daily.max()) / peak_grid)
        a_exact.append(float(pI) / peak_exact)
        C_obs.append(c_obs)
        C_end.append(c_end)

    dt_grid = np.asarray(dt_grid); dt_exact = np.asarray(dt_exact)
    a_grid = np.asarray(a_grid);   a_exact = np.asarray(a_exact)
    C_end = np.asarray(C_end, dtype=float)
    C_obs = np.asarray(C_obs, dtype=float)

    major = C_end >= MAJOR_FRACTION * C_inf
    major_obs = C_obs >= MAJOR_FRACTION * C_inf

    # The summary rows below keep only means and standard deviations, but the
    # generator's empirical scaling route needs the scaling factors themselves.
    # Everything is kept rather than just the major-outbreak subset, so the
    # conditioning can be changed later without rerunning. The seed is a
    # deterministic function of the cell, so these are exactly the trajectories
    # the published summary numbers were computed from.
    #
    # Written as CSV rather than a numpy archive so that anyone can open one
    # without knowing numpy's conventions -- these go in the public repository.
    # alpha, beta and R0 are repeated on every row so a file stands on its own
    # and the sixteen can be concatenated directly, which is how the coverage
    # results are already handled.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = ("samples_N%d_alpha%s_beta%s.csv"
                % (N, str(alpha).replace('.', 'p'), str(beta_base).replace('.', 'p')))
        pd.DataFrame(dict(
            alpha=alpha, beta=beta, N=N, R0=R0,
            trajectory=np.arange(len(dt_grid)),
            dt_grid=dt_grid, dt_exact=dt_exact,
            a_grid=a_grid, a_exact=a_exact,
            major=major, C_end=C_end, C_obs=C_obs,
        )).to_csv(os.path.join(save_dir, name), index=False)

    row = dict(alpha=alpha, beta_base=beta_base, beta_scaled=beta,
               N=N, R0=R0, C_inf_det=C_inf, n_traj=n_traj, seed=seed,
               n_major=int(major.sum()), major_frac=float(major.mean()),
               major_frac_at_obs=float(major_obs.mean()),
               survival_theory=1.0 - (1.0 / R0) ** I0,
               mean_C_major=float(C_end[major].mean()) if major.any() else np.nan,
               frac_unfinished=float((C_end > C_obs).mean()))
    for tag, m in (("all_grid", moments(dt_grid, a_grid)),
                   ("all_exact", moments(dt_exact, a_exact)),
                   ("major_grid", moments(dt_grid[major], a_grid[major])),
                   ("major_exact", moments(dt_exact[major], a_exact[major]))):
        for k, v in m.items():
            row[k + "_" + tag] = v
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True, help="params_combos.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--trajectories", type=int, default=1000)
    p.add_argument("--populations", default="100,1000,10000")
    p.add_argument("--save-samples", default=None,
                   help="directory for the per-trajectory shifts and amplitudes; "
                        "omit to write only the summary rows")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = pd.read_csv(args.params)

    for N in [int(x) for x in args.populations.split(",")]:
        rows = []
        for idx, prow in params.iterrows():
            # seed is a deterministic function of the cell, so any single cell
            # can be re-run on its own and reproduce exactly
            seed = 20261113 + 1000 * N + idx
            r = run_cell(float(prow["Alpha"]), float(prow["Beta"]), N,
                         args.trajectories, seed, save_dir=args.save_samples)
            rows.append(r)
            print("N=%-6d alpha=%-5g beta=%-7g R0=%6.2f  major %.3f (theory %.3f)"
                  "  sd_dt %.3f -> %.3f  sd_a %.4f"
                  % (N, r["alpha"], r["beta_base"], r["R0"], r["major_frac"],
                     r["survival_theory"], r["std_dt_major_grid"],
                     r["std_dt_major_exact"], r["std_a_major_exact"]), flush=True)
        out = os.path.join(args.out, "summary_N%d.csv" % N)
        pd.DataFrame(rows).to_csv(out, index=False)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
