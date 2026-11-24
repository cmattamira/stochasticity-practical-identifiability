"""Appendix table of extinction probabilities.

For each distinct R0 in the parameter grid, prints the branching-process
probability of early extinction, (1/R0)^I0, and the fraction of simulated CTMC
realisations classified as major outbreaks by the rule used throughout the
paper: final size at least a quarter of the deterministic final size.

The simulator, the final-size calculation and the rule are imported from
regen_best_fit.py, so this table uses exactly the code that produced the
fitted pairs.

    python extinction_table.py --params ../data/params_combos.csv
"""

import argparse
import random

import pandas as pd

import regen_best_fit as rb


def major_fraction(alpha, beta, n_realisations, seed):
    """Fraction of n_realisations whose final size reaches the cutoff."""
    rng = random.Random(seed)
    cutoff = rb.MAJOR_FRACTION * rb.deterministic_final_size(alpha, beta)
    n_major = 0
    for _ in range(n_realisations):
        daily, final_size = rb.simulate(alpha, beta, rng)
        if final_size >= cutoff:
            n_major = n_major + 1
    return n_major / n_realisations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True, help="params_combos.csv")
    p.add_argument("--seed", type=int, default=20261124)
    args = p.parse_args()

    params = pd.read_csv(args.params)
    params["R0"] = params["Beta"] * rb.N0 / params["Alpha"]
    params = params.sort_values("R0")

    print("%6s %6s %7s  %10s  %10s  %10s  %6s"
          % ("alpha", "beta", "R0", "P(extinct)", "P(major)", "simulated", "n"))
    for _, row in params.iterrows():
        alpha, beta, R0 = float(row["Alpha"]), float(row["Beta"]), float(row["R0"])
        # More realisations where the answer is far from 0 or 1.
        if R0 < 2:
            n = 5000
        elif R0 < 3:
            n = 1500
        else:
            n = 1000
        p_extinct = (1.0 / R0) ** rb.I0
        simulated = major_fraction(alpha, beta, n, args.seed)
        print("%6.2f %6.4f %7.2f  %10.3f  %10.3f  %10.3f  %6d"
              % (alpha, beta, R0, p_extinct, 1 - p_extinct, simulated, n), flush=True)


if __name__ == "__main__":
    main()
