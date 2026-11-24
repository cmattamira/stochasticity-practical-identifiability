"""Time needed to generate synthetic datasets: direct CTMC simulation against
the warped ODE, with the fitting step excluded.

For each population size N the script generates datasets at one parameter
combination by both methods and reports the time per 1000 datasets and the
ratio of the two. The transmission rate is rescaled with N so that R0 stays
fixed, as in fig7_regen.py.

The CTMC is the event-by-event Gillespie simulator in fig7_regen.py, so its
cost grows with the number of events and hence with N. The warped ODE solves
the deterministic model once and evaluates it at shifted times, so its cost
does not depend on N.

    python timing_benchmark.py --summary ../data/fig7/summary_N1000.csv
"""

import argparse
import random
import time

import numpy as np

from fig7_regen import simulate, I0
from time_shift import draw_time_shift
from peak_scaling import load_scaling_moments, draw_scaling_factor
from warp import ode_prevalence_extended, warp_exact

T_OBS = 100
t_eval = np.linspace(1.0, float(T_OBS), T_OBS)


def time_ctmc(alpha, beta, N, n_datasets, seed):
    """Seconds to simulate n_datasets CTMC realisations."""
    rng = random.Random(seed)
    start = time.perf_counter()
    for _ in range(n_datasets):
        simulate(alpha, beta, N, rng)
    return time.perf_counter() - start


def time_warp(alpha, beta, N, n_datasets, moments, seed):
    """Seconds to generate n_datasets warped-ODE datasets."""
    np.random.seed(seed)
    init_cond = np.array([N - I0, I0, 0, 0])
    start = time.perf_counter()
    delta_t = draw_time_shift(alpha, beta, N, I0, size=n_datasets)
    # The scaling moments were measured at N = 1000. Which N they are read at
    # does not affect the time taken, which is all this script measures.
    a = draw_scaling_factor(alpha, beta, N, size=n_datasets, moments_by_R0=moments)
    pad = float(np.abs(delta_t).max()) + 1.0
    prevalence = ode_prevalence_extended(alpha, beta, init_cond, t_eval, pad)
    for m in range(n_datasets):
        warp_exact(prevalence, t_eval, delta_t[m], a[m])
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, help="summary_N1000.csv")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.0004, help="value at N = 1000")
    p.add_argument("--populations", default="1000,10000,100000")
    p.add_argument("--seed", type=int, default=20261124)
    args = p.parse_args()

    moments = load_scaling_moments(args.summary)
    print("alpha = %g, beta = %g at N = 1000 (R0 = %.2f)"
          % (args.alpha, args.beta, args.beta * 1000 / args.alpha))
    print("%8s  %18s  %18s  %8s" % ("N", "CTMC s per 1000", "warped s per 1000", "ratio"))
    for N in [int(x) for x in args.populations.split(",")]:
        beta = args.beta * 1000 / N          # hold R0 fixed as N changes
        # Fewer CTMC realisations at large N, where each one is slow; the
        # result is scaled to 1000 datasets.
        n_ctmc = max(10, int(1000 * 1000 / N))
        ctmc = time_ctmc(args.alpha, beta, N, n_ctmc, args.seed) * 1000 / n_ctmc
        warped = time_warp(args.alpha, beta, N, 1000, moments, args.seed)
        print("%8d  %18.2f  %18.2f  %8.1f" % (N, ctmc, warped, ctmc / warped), flush=True)


if __name__ == "__main__":
    main()
