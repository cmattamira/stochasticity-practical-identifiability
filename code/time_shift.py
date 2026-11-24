"""Theory-based time shift for the warped-ODE generator.

Conditional on a major outbreak, a Markovian SIR epidemic in its early phase is
a linear birth-death process with birth rate beta*N and death rate alpha. The
martingale limit W = lim I(t) exp(-r t) exists, and the randomness in W appears
in the macroscopic phase as a pure translation of the deterministic trajectory:

    I(t) ~= I_ODE(t - Delta_t),     Delta_t = -(log W - log I0) / r

The centring is on log I0 because that is where the deterministic solution
starts, not because it is a sample mean: I_ODE(t) ~= I0 exp(r t) early on, so
matching I(t) ~= W exp(r t) to it gives exactly this shift. Dropping the log I0
term moves every shift by log(I0)/r days, which is six days at r = 0.2 and is
invisible in the variance.

With I0 initial infectives, each line survives the branching phase independently
with probability p = 1 - 1/R0. Conditional on the outbreak taking off at all,
the number of surviving lines m is Binomial(I0, p) restricted to m >= 1, and

    W | m ~ Gamma(shape = m, rate = p)

so W is a Gamma mixture rather than the single exponential of the I0 = 1 case.

Draws come from numpy's legacy global generator, which is what the rest of the
coverage code uses and what np.random.seed() in the script headers seeds.

References: Barbour, Hamza, Kaspi & Klebaner (2015); Barbour, Chigansky &
Klebaner (2016); Morris, Maclean & Black (2024).
"""

import numpy as np


class SubcriticalParameters(ValueError):
    """Raised for a parameter pair with R0 <= 1.

    There is no major outbreak to shift, and r = beta*N - alpha is non-positive
    so the shift would change sign. Callers should catch this and count it
    rather than substituting a value, so that we can see how often fitted pairs
    stray below threshold instead of hiding it.
    """


def shift_parameters(alpha, beta, N0):
    """Basic reproduction number, early growth rate, and survival probability."""
    R0 = beta * N0 / alpha
    if R0 <= 1.0:
        raise SubcriticalParameters(
            "R0 = %.6g at alpha = %.6g, beta = %.6g, N0 = %g" % (R0, alpha, beta, N0))
    r = beta * N0 - alpha
    p = 1.0 - 1.0 / R0
    return R0, r, p


def draw_time_shift(alpha, beta, N0, I0, size=None):
    """Draw the time shift from the branching-process limit.

    Returns a scalar if size is None, otherwise an array of that length.
    Positive Delta_t means the epidemic runs late relative to the ODE, so the
    warped curve is I_ODE(t - Delta_t).
    """
    R0, r, p = shift_parameters(alpha, beta, N0)

    n = 1 if size is None else int(size)

    # Surviving lines among the I0 initial infectives, conditioned on at least
    # one surviving. Rejection is simpler to read than building the truncated
    # pmf, and P(m = 0) = (1 - p)^I0 = (1/R0)^I0 is small except at low R0.
    m = np.random.binomial(I0, p, size=n)
    while True:
        extinct = (m == 0)
        if not extinct.any():
            break
        m[extinct] = np.random.binomial(I0, p, size=int(extinct.sum()))

    # numpy's gamma takes a scale, so pass 1/p for a rate of p.
    W = np.random.gamma(shape=m, scale=1.0 / p)

    delta_t = -(np.log(W) - np.log(I0)) / r
    return float(delta_t[0]) if size is None else delta_t


def shift_moments(alpha, beta, N0, I0):
    """Exact mean and sd of Delta_t, for checking the sampler.

    E[log W | m] = psi(m) - log p and Var(log W | m) = psi'(m), so over the
    conditioned Binomial mixture the law of total variance gives

        Var(log W) = E[psi'(m)] + Var(psi(m))

    At I0 = 1 this reduces to psi'(1) = pi^2/6, independent of p, which is the
    classical Gumbel result for the exponential case.
    """
    from scipy.special import digamma, polygamma
    from scipy.stats import binom

    R0, r, p = shift_parameters(alpha, beta, N0)

    m = np.arange(1, I0 + 1)
    w = binom.pmf(m, I0, p)
    w = w / w.sum()                      # renormalise over m >= 1

    psi = digamma(m)
    psi1 = polygamma(1, m)

    mean_log_W = np.sum(w * psi) - np.log(p)
    var_log_W = np.sum(w * psi1) + (np.sum(w * psi ** 2) - np.sum(w * psi) ** 2)

    mean_shift = -(mean_log_W - np.log(I0)) / r
    sd_shift = np.sqrt(var_log_W) / abs(r)
    return mean_shift, sd_shift
