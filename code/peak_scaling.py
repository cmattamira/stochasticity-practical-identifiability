"""Peak scaling factor for the warped-ODE generator.

The warped trajectory is

    I_warped(t) = a * I_ODE(t - Delta_t)

Piece 1 (time_shift.py) draws Delta_t from the branching-process limit, with no
calibration at all. This module draws a, and it cannot be done the same way:
the limit theorems give a pure time shift and say nothing about the peak
height. The scaling factor is a finite-population effect, so it has to be
measured.

Two ways of drawing it, matching the two routes in the generator spec:

  empirical    resample from the scaling factors measured in CTMC runs at the
               true parameters of one cell. This is what the published code
               does. It does not adapt to the fitted pair, which is the
               circularity described in case4.py.

  parametric   read a mean and standard deviation off the measured values as
               functions of R0, evaluated at the fitted pair's own R0, and draw
               a normal with those moments. This adapts to the fitted pair, so
               the circularity is reduced to interpolation across the parameter
               grid. It is not parameter-free in the way the time shift is, and
               should not be described as such.

Which one you get is decided by which calibration you pass in -- see
draw_scaling_factor. Both are built from the same simulations; they differ in
how much is kept. The empirical route keeps every measured value from one cell;
the parametric route keeps two numbers from each of sixteen cells, which is what
makes interpolation in R0 possible.

Why a normal. The measured scaling factors are close to symmetric: skewness is
within +/- 0.11 in twelve of the sixteen cells, and an Anderson-Darling test for
normality is below the 5% critical value in thirteen. A lognormal, which would
be the obvious guess for a positive ratio, fits worse in most cells. The one
clear exception is R0 = 1.21, where the distribution is genuinely skewed and
nothing simple fits; that cell is outside the range where any of this works.

Draws come from numpy's legacy global generator, matching time_shift.py and the
rest of the coverage code.
"""

import os

import numpy as np

from time_shift import shift_parameters, shift_moments


def load_scaling_moments(summary_csv, population=1000):
    """Mean and standard deviation of the scaling factor against R0.

    Reads the Fig 7 summary written by fig7_regen.py and keeps the rows for one
    population size. The coverage protocol runs entirely at N = 1000, so there
    is no need to model how these vary with N -- a single column is enough, and
    R0 is the only thing to interpolate in.

    Where two parameter combinations share an R0 -- there are four such pairs in
    the grid -- their values are averaged. They differ only in alpha, and the
    two members of each pair agree closely, so averaging loses nothing and keeps
    R0 strictly increasing, which the interpolation needs.

    Returns a dictionary of three arrays, sorted by R0.
    """
    import pandas as pd

    d = pd.read_csv(summary_csv)
    d = d[d["N"] == population]
    if len(d) == 0:
        raise ValueError("no rows with N = %d in %s" % (population, summary_csv))

    g = d.groupby(d["R0"].round(6)).agg(
        mean=("mean_a_major_exact", "mean"),
        sd=("std_a_major_exact", "mean")).reset_index()
    g = g.sort_values("R0")

    return dict(R0=g["R0"].to_numpy(),
                mean=g["mean"].to_numpy(),
                sd=g["sd"].to_numpy())


def scaling_moments(alpha, beta, N0, moments_by_R0):
    """Mean and sd of the scaling factor at this parameter pair.

    Mirrors shift_moments in time_shift.py, which returns the same two numbers
    for the time shift. The difference between them is worth noticing: the shift
    needs no calibration argument because it comes from theory, while this one
    does because it is measured.

    Both quantities are interpolated against R0 on log scales, because both span
    more than an order of magnitude across the grid and are close to straight
    lines there. The mean is interpolated as mean - 1, which is the part that
    actually varies; the mean itself is near 1 everywhere above R0 = 2 and
    interpolating it directly would lose most of the signal to rounding.

    Outside the measured range the end values are held constant rather than
    extrapolated. Above R0 = 14.29 the sd is already 0.021 and still falling, so
    holding it flat overstates the spread very slightly. Below R0 = 1.21 nothing
    about this method works anyway.
    """
    R0, _, _ = shift_parameters(alpha, beta, N0)

    log_R0 = np.log(moments_by_R0["R0"])
    mean = 1.0 + np.exp(np.interp(np.log(R0), log_R0,
                                  np.log(moments_by_R0["mean"] - 1.0)))
    sd = np.exp(np.interp(np.log(R0), log_R0, np.log(moments_by_R0["sd"])))
    return float(mean), float(sd)


def load_peak_scaling_factors(samples_dir, alpha, beta, peak="exact"):
    """The measured scaling factors for one cell, major outbreaks only.

    These come from the CSV files fig7_regen.py writes with --save-samples, one
    row per trajectory.

    peak = "exact" uses the true continuous peak of each trajectory; "grid"
    uses the largest daily report, which is what the published code used. The
    grid value is the smaller of the two whenever the peak falls between
    reports, so it understates the height and, taken as a ratio to the ODE's
    own grid peak, carries that bias in both numerator and denominator.
    """
    import pandas as pd

    name = ("samples_N1000_alpha%s_beta%s.csv"
            % (str(alpha).replace('.', 'p'), str(beta).replace('.', 'p')))
    d = pd.read_csv(os.path.join(samples_dir, name))
    return d.loc[d["major"], "a_" + peak].to_numpy()


def draw_scaling_factor(alpha, beta, N0, size=None, moments_by_R0=None,
                        peak_scaling_factors=None, rho=0.0, delta_t=None,
                        I0=5, max_tries=100):
    """Draw the peak scaling factor.

    Pass moments_by_R0 for the parametric route, or peak_scaling_factors for the
    empirical one. Exactly one of them, so the call site says which is in use.

    alpha and beta are always the FITTED pair. Under the parametric route they
    set the moments, which is the point of it. Under the empirical route they
    are not used at all: the values passed in were measured at some cell's true
    parameters and are resampled whatever the fitted pair happens to be. That
    mismatch is the circularity, and it is left visible here rather than hidden.

    rho couples the scaling factor to the time shift, for the sensitivity check.
    The default of zero draws them independently, which is what the published
    coverage code did. Passing a non-zero rho requires delta_t, the shift
    already drawn for this realisation, and produces a scaling factor with that
    linear correlation to it while keeping the same mean and sd. Only the
    parametric route supports it; resampled values have no such handle.

    A normal can go below zero, which would turn the trajectory upside down.
    Anywhere above R0 = 2 zero is more than twenty standard deviations away and
    this never happens; at R0 = 1.21 it is under three, so about one draw in
    three hundred needs replacing. Redrawing is the same approach time_shift.py
    takes for the extinct branching lines.
    """
    if (moments_by_R0 is None) == (peak_scaling_factors is None):
        raise ValueError("pass either moments_by_R0 or peak_scaling_factors, "
                         "not both and not neither")

    n = 1 if size is None else int(size)

    if peak_scaling_factors is not None:
        if rho != 0.0:
            raise ValueError("rho only applies to the parametric route")
        a = np.random.choice(np.asarray(peak_scaling_factors), size=n,
                             replace=True)
        return float(a[0]) if size is None else a

    mean, sd = scaling_moments(alpha, beta, N0, moments_by_R0)

    if rho == 0.0:
        z = np.random.normal(0.0, 1.0, size=n)
    else:
        if delta_t is None:
            raise ValueError("a non-zero rho needs the delta_t drawn with it")
        # Standardise the shift against its own exact moments, so that mixing it
        # in changes the correlation without changing the scaling factor's mean
        # or sd. The shift is not itself normal, but with rho near -0.1 it
        # contributes about 1% of the variance, so the result stays normal to
        # any accuracy that matters here.
        mean_dt, sd_dt = shift_moments(alpha, beta, N0, I0)
        z_dt = (np.asarray(delta_t, dtype=float) - mean_dt) / sd_dt
        z = rho * z_dt + np.sqrt(1.0 - rho ** 2) * np.random.normal(0.0, 1.0, size=n)

    a = mean + sd * z

    # Replace any non-positive draw rather than clipping it, so the kept values
    # are a sample from the truncated normal instead of a spike at zero.
    for _ in range(max_tries):
        bad = a <= 0.0
        if not bad.any():
            break
        a[bad] = mean + sd * np.random.normal(0.0, 1.0, size=int(bad.sum()))
    else:
        raise RuntimeError("could not draw a positive scaling factor at "
                           "alpha = %g, beta = %g" % (alpha, beta))

    return float(a[0]) if size is None else a
