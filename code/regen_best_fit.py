"""Regenerate best_fit_params with major-outbreak conditioning.

Steps 1 and 2 of the coverage protocol: simulate J realisations at the true
parameters, fit the ODE to each, and save the J fitted pairs. Every later step
reads these files.

Why: the existing files were produced without conditioning, so realisations that
fizzled were fitted anyway. Fitting an SIR curve to a trajectory that never took
off has no sensible answer, and Nelder-Mead converges happily to nonsense --
beta of order 1e-9, or alpha of order 1e-7 giving R0 = 94,093. Seventeen of the
hundred pairs at R0 = 1.21 and two at R0 = 2.00 are like that, and they then
enter the bootstrap as though they were legitimate fits. The optimiser's exit
status cannot catch them: all of them report success, because they ARE the best
fit to data that carries no information about beta.

A realisation counts as a major outbreak if its final size reaches a quarter of
the deterministic final size for that parameter pair, C_inf = S0 - S_inf with
S_inf solving S_inf = S0 exp(-R0 (N - S_inf)/N). For R0 >= 2 the classification
is identical for any threshold between a tenth and nine tenths of C_inf, and at
R0 = 1.21 this choice reproduces the branching-process survival probability
1 - R0^(-I0) to within 0.008.

Fitting matches the existing convention exactly, so old and new numbers stay
comparable: unweighted sum of squares on prevalence, Nelder-Mead, started at the
true parameters, np.abs on the result.


    python regen_best_fit.py --params params_combos.csv \\
        --out best_fit_params_conditioned [--runs 100]
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize

N0 = 1000
I0 = 5
T_OBS = 100
MAJOR_FRACTION = 0.25
T_CAP = 5000.0
EVENT_CAP = 50_000_000

t_eval = np.linspace(1.0, float(T_OBS), T_OBS)
init_cond = np.array([N0 - I0, I0, 0, 0])


def SIR_model(t, y, alpha, beta):
    S, I, R, C = y
    return [-beta * S * I, beta * S * I - alpha * I, alpha * I, beta * S * I]


def SolveModel(par, tspan, y0, te):
    return solve_ivp(SIR_model, tspan, y0, args=tuple(par), t_eval=te,
                     method="RK45")


def objective_fun(par, tdata, ydata, ic):
    I = SolveModel(par, (tdata[0], tdata[-1]), ic, tdata).y[1]
    return np.sum((I - ydata) ** 2)


def deterministic_final_size(alpha, beta):
    """Cumulative new infections over the whole epidemic, from the ODE."""
    R0 = beta * N0 / alpha
    S0 = N0 - I0
    f = lambda S: S - S0 * np.exp(-R0 * (N0 - S) / N0)
    return S0 - brentq(f, 1e-12, S0 - 1e-12, xtol=1e-12, rtol=1e-14)


def simulate(alpha, beta, rng):
    """One realisation: daily prevalence, and cumulative incidence at the end."""
    S, I, C = N0 - I0, I0, 0
    t, nxt = 1.0, 1
    daily = np.zeros(T_OBS)
    events = 0
    while I > 0 and t < T_CAP and events < EVENT_CAP:
        rate_inf = beta * S * I
        rate_rec = alpha * I
        total = rate_inf + rate_rec
        t_next = t - math.log(rng.random()) / total
        while nxt <= T_OBS and t_next > nxt:
            daily[nxt - 1] = I
            nxt += 1
        t = t_next
        events += 1
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


def run_cell(alpha, beta, n_runs, seed):
    C_inf = deterministic_final_size(alpha, beta)
    cutoff = MAJOR_FRACTION * C_inf
    rng = random.Random(seed)

    fitted = np.zeros((2, n_runs))
    rejected = 0
    i = 0
    while i < n_runs:
        daily, C_end = simulate(alpha, beta, rng)
        if C_end < cutoff:
            rejected += 1
            continue
        res = minimize(objective_fun, [alpha, beta],
                       args=(t_eval, daily, init_cond), method="Nelder-Mead")
        fitted[:, i] = np.abs(res.x)
        i += 1

    R0_fit = fitted[1] * N0 / fitted[0]
    return fitted, dict(
        alpha=alpha, beta=beta, R0=beta * N0 / alpha, C_inf=C_inf,
        cutoff=cutoff, n_runs=n_runs, rejected=rejected, seed=seed,
        accept_rate=n_runs / (n_runs + rejected),
        survival_theory=1.0 - (1.0 / (beta * N0 / alpha)) ** I0,
        R0_min=float(R0_fit.min()), R0_max=float(R0_fit.max()),
        R0_median=float(np.median(R0_fit)),
        n_degenerate=int(((R0_fit <= 1.0) | (R0_fit > 50.0)).sum()),
        alpha_median=float(np.median(fitted[0])),
        beta_median=float(np.median(fitted[1])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--compare", default=None,
                   help="existing best_fit_params directory, for the report")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = pd.read_csv(args.params)
    report = []

    for idx, row in params.iterrows():
        alpha = float(row["Alpha"])
        beta = float(row["Beta"])
        seed = 20261113 + idx
        fitted, info = run_cell(alpha, beta, args.runs, seed)

        fname = ("best_par_alpha%s_beta%s.csv"
                 % (str(alpha).replace(".", "p"), str(beta).replace(".", "p")))
        np.savetxt(os.path.join(args.out, fname), fitted, delimiter=",")

        if args.compare:
            old_path = os.path.join(args.compare, fname)
            if os.path.exists(old_path):
                old = np.loadtxt(old_path, delimiter=",")
                r0_old = old[1] * N0 / old[0]
                info["old_degenerate"] = int(((r0_old <= 1.0) | (r0_old > 50.0)).sum())
                info["old_R0_median"] = float(np.median(r0_old))
                info["old_R0_min"] = float(r0_old.min())
                info["old_R0_max"] = float(r0_old.max())
        report.append(info)
        print("R0=%6.2f  accepted %d, rejected %3d (accept %.3f, theory %.3f) | "
              "fitted R0 %.3f .. %.3f, median %.3f | degenerate %d%s"
              % (info["R0"], info["n_runs"], info["rejected"],
                 info["accept_rate"], info["survival_theory"],
                 info["R0_min"], info["R0_max"], info["R0_median"],
                 info["n_degenerate"],
                 "" if "old_degenerate" not in info
                 else "  (was %d)" % info["old_degenerate"]), flush=True)

    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(args.out, "regeneration_report.csv"), index=False)
    print("\nwrote %d files plus regeneration_report.csv to %s"
          % (len(report), args.out))
    print("degenerate pairs: %d now" % rep["n_degenerate"].sum(), end="")
    if "old_degenerate" in rep:
        print(", %d before" % rep["old_degenerate"].sum())
    else:
        print()


if __name__ == "__main__":
    main()
