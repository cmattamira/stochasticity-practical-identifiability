"""Is the KDE confidence region hitting its nominal level?

Diagnostic only. Nothing here goes into the paper; it tests the estimator that
produces every coverage number we quote.

compute_2d_confidence_region() builds a 68% region by fitting a Gaussian kernel
density to the bootstrap cloud, evaluating it on a grid, and keeping the
highest-density grid cells until they hold 68% of the summed density. Two
choices in that recipe could shift the level away from 68% on their own, with
no epidemic involved:

  1. Scott's bandwidth. gaussian_kde's default widens the estimated density
     relative to the truth. A wider density gives a wider 68% region, and a
     wider region covers the true point more often than it should.

  2. The grid. It spans only x.min() to x.max() and y.min() to y.max(), so any
     density mass outside the sample's own range is left out of the sum. The
     68% threshold is then taken against a total that is already short of 1.

The test: draw samples from a bivariate normal whose mean we know, so the
correct answer is exactly 0.68 by construction, and ask what the function
actually returns. Anything well away from 0.68 is a bias carried by every cell
of the coverage table, in all three arms, published and revised alike.

A fourth experiment adds a few degenerate points to an otherwise clean sample.
That is the mechanism proposed for the case3c result at R0 = 4.00, where
removing roughly one bootstrap replicate per thousand moved coverage by 0.09.

    python kde_calibration.py
"""

import numpy as np
from scipy.stats import gaussian_kde

CONFIDENCE_LEVEL = 0.68


def region_contains(x, y, true_point, confidence_level=CONFIDENCE_LEVEL,
                    pad=0.0, bw_method=None):
    """True if true_point falls inside the KDE confidence region.

    With pad = 0 and bw_method = None this is exactly the body of
    compute_2d_confidence_region's loop, copied so the two cannot drift apart.
    pad widens the grid by that fraction of the sample range on each side, and
    bw_method is passed straight to gaussian_kde; both are here only so the two
    suspect choices can be switched off one at a time.
    """
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)

    x_lo, x_hi = x.min(), x.max()
    y_lo, y_hi = y.min(), y.max()
    if pad > 0:
        x_lo, x_hi = x_lo - pad * (x_hi - x_lo), x_hi + pad * (x_hi - x_lo)
        y_lo, y_hi = y_lo - pad * (y_hi - y_lo), y_hi + pad * (y_hi - y_lo)

    X, Y = np.meshgrid(np.linspace(x_lo, x_hi, 100),
                       np.linspace(y_lo, y_hi, 100))
    Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])), X.shape)

    sorted_Z = np.sort(Z.ravel())[::-1]
    cumsum_Z = np.cumsum(sorted_Z) / np.sum(sorted_Z)
    level_idx = np.argmax(cumsum_Z >= confidence_level)
    level_val = sorted_Z[level_idx]

    return kde(true_point) >= level_val


def draw_cloud(n, rho, rng):
    """One bootstrap cloud, in the idealised case where the answer is known.

    The cloud is NOT centred on the true parameters. It is centred on a fitted
    pair, which is itself one draw away from the truth -- that is the whole
    reason coverage is not 1. So each repetition does what the pipeline does:

        true parameters      the origin, fixed
        fitted pair          one draw from N(truth, Sigma)       <- step 1
        bootstrap cloud      n draws from N(fitted pair, Sigma)  <- step 3

    With a normal model and the same Sigma at both stages, the truth is a draw
    from the same distribution as the cloud points, so a correctly calibrated
    68% region contains it 68% of the time. Any other answer is the estimator.

    (Centring the cloud on the truth instead gives coverage 1.000, since the
    highest-density region always contains the mode. That was the first version
    of this script and the result was meaningless.)
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    fitted_pair = rng.multivariate_normal([0.0, 0.0], cov)
    return rng.multivariate_normal(fitted_pair, cov, size=n).T


def add_degenerate(x, y, n_bad, rng):
    """Replace n_bad of the points with fits that have collapsed.

    A failed Nelder-Mead fit in the real pipeline returns beta of order 1e-9 or
    alpha of order 1e-7 -- values hundreds of sample standard deviations from
    the cloud. Here that is imitated by moving a few points far out along one
    axis, keeping the rest of the sample untouched.
    """
    x, y = x.copy(), y.copy()
    which = rng.choice(len(x), size=n_bad, replace=False)
    x[which] = x[which] - 50.0
    y[which] = y[which] - 50.0
    return x, y


def coverage(n_rep, n, rho, pad=0.0, bw_method=None, n_bad=0, seed=0):
    """Fraction of repetitions whose region contains the true point."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_rep):
        x, y = draw_cloud(n, rho, rng)
        if n_bad > 0:
            x, y = add_degenerate(x, y, n_bad, rng)
        hits += bool(region_contains(x, y, [0.0, 0.0], pad=pad,
                                     bw_method=bw_method))
    return hits / n_rep


def standard_error(p, n_rep):
    return np.sqrt(p * (1 - p) / n_rep)


def report(label, p, n_rep):
    print(f"  {label:<46s} {p:.3f}  +/- {standard_error(p, n_rep):.3f}")


if __name__ == "__main__":

    N_REP = 2000          # Monte Carlo error on each number below is ~0.010
    N = 1000              # bootstrap replicates per fitted pair, as we run it

    print("\nNominal level is 0.68 in every line. Everything is Gaussian and")
    print("correctly specified, so any departure from 0.68 is the estimator's.\n")

    print("1. As the pipeline runs it, independent coordinates")
    p = coverage(N_REP, N, rho=0.0, seed=1)
    report(f"n = {N}, Scott bandwidth, grid = sample range", p, N_REP)

    print("\n2. The same, with the (alpha, beta) correlation the fits show")
    for rho in (0.9, 0.99):
        p = coverage(N_REP, N, rho=rho, seed=2)
        report(f"n = {N}, correlation {rho}", p, N_REP)

    print("\n3. The two suspect choices, switched off one at a time")
    p = coverage(N_REP, N, rho=0.0, pad=0.5, seed=3)
    report("grid padded 50% beyond the sample range", p, N_REP)
    for factor in (0.5, 0.75):
        p = coverage(N_REP, N, rho=0.0, bw_method=factor * N ** (-1.0 / 6.0),
                     seed=4)
        report(f"bandwidth at {factor} of Scott's", p, N_REP)
    p = coverage(N_REP, N, rho=0.0, pad=0.5,
                 bw_method=0.5 * N ** (-1.0 / 6.0), seed=5)
    report("both: padded grid and half bandwidth", p, N_REP)

    print("\n4. Sample size, at the two counts we have run")
    for n in (100, 1000):
        p = coverage(N_REP, n, rho=0.0, seed=6)
        report(f"n = {n}", p, N_REP)

    print("\n5. A few collapsed fits in an otherwise clean sample")
    print("   (this is the case3c mechanism: n_bad = 1 is one in a thousand)")
    for n_bad in (0, 1, 2, 5, 20):
        p = coverage(N_REP, N, rho=0.0, n_bad=n_bad, seed=7)
        report(f"{n_bad} degenerate points out of {N}", p, N_REP)

    print()
