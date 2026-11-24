"""Coverage from the same bootstrap clouds, three ways.

The coverage numbers in the paper come from one confidence-region estimator:
a Gaussian kernel density on the bootstrap cloud, evaluated on a grid, with
the 68% highest-density cells kept. That estimator turns out to over-cover by
about 0.03 on correctly specified Gaussian data, and a single collapsed fit in
a thousand adds a further 0.065 (see KDE_Calibration.md).

Neither of those threatens the comparison between arms, because all three arms
share the estimator and the bias is common to them all. This script is the
evidence for that claim: it recomputes coverage from the saved clouds under
three different rules and asks whether the gap between arms survives.

    KDE          the paper's rule, copied unchanged, as the reference point.
    ellipse      Fit a mean and covariance to the cloud and keep the ellipse
                 holding 68% of a bivariate normal. No bandwidth, no grid, but
                 it assumes the cloud is elliptical.
    rank         Measure every bootstrap point's Mahalanobis distance from the
                 cloud centre, and call the truth covered if its own distance
                 falls below the 68th percentile of those distances. This is a
                 rank, so it is calibrated by construction whenever the
                 bootstrap is consistent, whatever the cloud's shape.

If all three put the arms in the same order with similar gaps, the result does
not depend on the estimator. If they disagree, that needs to be known and
reported.

    python compare_estimators.py --clouds /path/to/clouds
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde

CONFIDENCE_LEVEL = 0.68


def covered_kde(x, y, true_point, level=CONFIDENCE_LEVEL):
    """The paper's rule, copied from compute_2d_confidence_region unchanged."""
    kde = gaussian_kde(np.vstack([x, y]))
    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                       np.linspace(y.min(), y.max(), 100))
    Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])), X.shape)

    sorted_Z = np.sort(Z.ravel())[::-1]
    cumsum_Z = np.cumsum(sorted_Z) / np.sum(sorted_Z)
    level_idx = np.argmax(cumsum_Z >= level)
    level_val = sorted_Z[level_idx]

    return bool(kde(true_point) >= level_val)


def mahalanobis(points, centre, cov):
    """Squared Mahalanobis distance of each column of points from centre."""
    diff = points - centre[:, None]
    return np.einsum("ij,ji->i", diff.T, np.linalg.solve(cov, diff))


def covered_ellipse(x, y, true_point, level=CONFIDENCE_LEVEL):
    """Inside the 68% ellipse of a normal fitted to the cloud."""
    cloud = np.vstack([x, y])
    centre = cloud.mean(axis=1)
    cov = np.cov(cloud)
    d_true = mahalanobis(np.array(true_point)[:, None], centre, cov)[0]
    return bool(d_true <= chi2.ppf(level, df=2))


def covered_rank(x, y, true_point, level=CONFIDENCE_LEVEL):
    """Truth's distance below the 68th percentile of the cloud's distances."""
    cloud = np.vstack([x, y])
    centre = cloud.mean(axis=1)
    cov = np.cov(cloud)
    d_cloud = mahalanobis(cloud, centre, cov)
    d_true = mahalanobis(np.array(true_point)[:, None], centre, cov)[0]
    return bool(d_true <= np.quantile(d_cloud, level))


def coverage_all_ways(cloud, true_point):
    """Fraction of fitted pairs covered, under each of the three rules."""
    n_pairs = cloud.shape[2]
    hits = dict(KDE=0, ellipse=0, rank=0)
    for i in range(n_pairs):
        x, y = cloud[0, :, i], cloud[1, :, i]
        hits["KDE"] += covered_kde(x, y, true_point)
        hits["ellipse"] += covered_ellipse(x, y, true_point)
        hits["rank"] += covered_rank(x, y, true_point)
    return {k: v / n_pairs for k, v in hits.items()}


def true_parameters_from_name(name):
    """Pull alpha and beta back out of the file name the run wrote."""
    parts = os.path.basename(name).replace(".npy", "").split("_")
    alpha = float([p for p in parts if p.startswith("alpha")][0][5:].replace("p", "."))
    beta = float([p for p in parts if p.startswith("beta")][0][4:].replace("p", "."))
    sigma_parts = [p for p in parts if p.startswith("sigma")]
    sigma = sigma_parts[0][5:].replace("p", ".") if sigma_parts else ""
    return alpha, beta, sigma


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", required=True,
                    help="directory holding the per-arm cloud directories")
    ap.add_argument("--N0", type=int, default=1000)
    args = ap.parse_args()

    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(args.clouds, "*"))):
        if not os.path.isdir(arm_dir):
            continue
        arm = os.path.basename(arm_dir)
        for f in sorted(glob.glob(os.path.join(arm_dir, "*.npy"))):
            alpha, beta, sigma = true_parameters_from_name(f)
            cloud = np.load(f)
            cov = coverage_all_ways(cloud, [alpha, beta])
            rows.append(dict(arm=arm, R0=round(beta * args.N0 / alpha, 2),
                             alpha=alpha, sigma=sigma, pairs=cloud.shape[2],
                             replicates=cloud.shape[1], **cov))
            print(f"  done {arm} {os.path.basename(f)}")

    out = pd.DataFrame(rows).sort_values(["arm", "R0", "sigma"])
    pd.set_option("display.width", 200)
    print()
    print(out.to_string(index=False))
    out.to_csv(os.path.join(args.clouds, "estimator_comparison.csv"), index=False)
    print(f"\nwritten to {os.path.join(args.clouds, 'estimator_comparison.csv')}")
