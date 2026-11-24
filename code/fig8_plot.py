"""Figure 8: how much independent Gaussian noise is needed to reach nominal coverage.

Left panel   the smallest noise level sigma at which the Gaussian-noise method
             reaches the nominal coverage, for six parameter combinations
             spanning R0.
Right panel  the full coverage-against-sigma curve for one combination.

The coverage numbers are not computed here. They come from case1.py, run once
per (combination, sigma) by the array job in submit_fig8.slurm, which writes
one small CSV per task. This script only reads those files and draws them.

Negative simulated values are not truncated anywhere in the pipeline. With
multiplicative noise y = I * (1 + eps), eps ~ N(0, sigma^2), a simulated point
is negative with probability Phi(-1/sigma), and that fraction is reported next
to each minimum sigma.

    python fig8_plot.py --results ../results/fig8_M1000 ../results/case1 --out ../figures

Give every directory that holds coverage CSVs for the run you want to plot. If
the same (combination, sigma) appears in more than one directory, the first
directory listed wins.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

NOMINAL = 0.68

# The six combinations of the left panel, as (alpha, beta). Five share
# alpha = 0.14 and so vary only R0; the sixth is the alpha = 0.10, R0 = 4
# combination used for the identifiability comparison.
PANEL_A = [(0.14, 0.0004), (0.10, 0.0004), (0.14, 0.0008),
           (0.14, 0.0012), (0.14, 0.0016), (0.14, 0.0020)]

# The combination whose full curve is drawn in the right panel.
PANEL_B = (0.14, 0.0004)


def load_coverage(directories):
    """Read every coverage CSV in the given directories into one table."""
    frames = []
    for d in directories:
        for path in sorted(glob.glob(os.path.join(d, "coverage_combo*.csv"))):
            frames.append(pd.read_csv(path))
    if len(frames) == 0:
        raise SystemExit("no coverage_combo*.csv files found in: %s" % ", ".join(directories))
    table = pd.concat(frames, ignore_index=True)
    table["Alpha_true"] = table["Alpha_true"].round(4)
    table["Beta_true"] = table["Beta_true"].round(6)
    table["Sigma"] = table["Sigma"].round(4)
    # Keep the first occurrence of each (combination, sigma).
    table = table.drop_duplicates(subset=["Alpha_true", "Beta_true", "Sigma"], keep="first")
    return table.sort_values(["Alpha_true", "Beta_true", "Sigma"])


def curve_for(table, alpha, beta):
    """Sigma and coverage for one combination, sorted by sigma."""
    rows = table[(table["Alpha_true"] == round(alpha, 4)) &
                 (table["Beta_true"] == round(beta, 6))]
    return rows["Sigma"].to_numpy(), rows["Coverage"].to_numpy()


def minimum_sigma(sigma, coverage, nominal=NOMINAL):
    """Smallest sigma at which coverage reaches the nominal level.

    Found by linear interpolation between the last grid point below the
    nominal level and the first one at or above it. Returns nan when the
    curve never reaches the nominal level on the grid that was run.
    """
    for k in range(len(sigma)):
        if coverage[k] >= nominal:
            if k == 0:
                return float(sigma[0])
            fraction = (nominal - coverage[k - 1]) / (coverage[k] - coverage[k - 1])
            return float(sigma[k - 1] + fraction * (sigma[k] - sigma[k - 1]))
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="directories holding coverage_combo*.csv files")
    p.add_argument("--out", default=".")
    args = p.parse_args()

    table = load_coverage(args.results)

    # ---- left panel: minimum sigma for each of the six combinations
    rows = []
    for alpha, beta in PANEL_A:
        sigma, coverage = curve_for(table, alpha, beta)
        s_min = minimum_sigma(sigma, coverage)
        rows.append(dict(alpha=alpha, beta=beta, R0=round(beta * 1000 / alpha, 2),
                         sigma_min=s_min,
                         percent_negative=100 * norm.cdf(-1 / s_min) if s_min == s_min else float("nan"),
                         sigma_values_run=len(sigma)))
    summary = pd.DataFrame(rows).sort_values("R0")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(summary["R0"], summary["sigma_min"], color="tab:orange", marker="x", s=50)
    ax1.axhspan(0, 0.2, color="gray", alpha=0.2)
    ax1.set_xlabel("$R_0$", fontsize=13)
    ax1.set_ylabel(r"Minimum $\sigma$ for 68% Coverage", fontsize=13)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # ---- right panel: the full curve for one combination
    sigma, coverage = curve_for(table, *PANEL_B)
    ax2.plot(sigma, coverage, marker="o", color="tab:blue")
    ax2.axhline(NOMINAL, linestyle="--", color="gray", label="Target coverage")
    ax2.set_xlabel(r"$\sigma$", fontsize=13)
    ax2.set_ylabel("Coverage Probability", fontsize=13)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=2.5, w_pad=5.0)

    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "Fig8_regen.pdf"))
    fig.savefig(os.path.join(args.out, "Fig8_regen.png"), dpi=200)
    summary.to_csv(os.path.join(args.out, "Fig8_sigma_min.csv"), index=False)

    print(summary.round({"sigma_min": 3, "percent_negative": 2}).to_string(index=False))
    print("\nright panel, alpha = %g, beta = %g:" % PANEL_B)
    for s, c in zip(sigma, coverage):
        print("   sigma = %.1f   coverage = %.2f" % (s, c))
    print("\nwritten to %s: Fig8_regen.pdf/.png, Fig8_sigma_min.csv" % args.out)


if __name__ == "__main__":
    main()
