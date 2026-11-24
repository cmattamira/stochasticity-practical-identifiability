"""Apply a time shift and an amplitude factor to the deterministic trajectory.

The warped curve is

    I_warped(t) = a * I_ODE(t - Delta_t)

clipped at zero. Positive Delta_t translates the epidemic to the right, so the
warped peak lands at t_peak + Delta_t.

Two ways of evaluating I_ODE(t - Delta_t) are provided, because they differ at
the edges of the observation window and it is worth knowing by how much:

  "interp"  linear interpolation of a * I_ODE against abscissae t_eval +
            Delta_t, with fill_value="extrapolate". This is what the existing
            Fig 6 and Fig 9 code does, so it is here for comparability.

  "exact"   integrate the ODE over a window wide enough to cover t - Delta_t
            directly, forwards and backwards from the initial condition, and
            read the solver's own dense output. No extrapolation anywhere.

The two agree in the interior. They differ on the rising limb when Delta_t is
positive, because t_eval + Delta_t then starts after the first observation time
and "interp" has to extrapolate backwards along a curve that is growing
exponentially -- which linear extrapolation undershoots, often to a negative
value that the clip then flattens to zero.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def _rhs(t, y, alpha, beta):
    S, I, R, C = y
    return [-beta * S * I, beta * S * I - alpha * I, alpha * I, beta * S * I]


def ode_prevalence(alpha, beta, init_cond, t_eval, rtol=1e-4, atol=1e-6):
    """Prevalence of the deterministic model on t_eval."""
    sol = solve_ivp(_rhs, (t_eval[0], t_eval[-1]), init_cond, args=(alpha, beta),
                    t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    return sol.y[1]


def ode_prevalence_extended(alpha, beta, init_cond, t_eval, pad,
                            rtol=1e-4, atol=1e-6):
    """A callable giving prevalence on [t0 - pad, t_end + pad].

    The initial condition sits at t_eval[0], so the window is built by
    integrating forwards to t_end + pad and backwards to t0 - pad, and reading
    the solver's dense output on each side.
    """
    t0, t1 = float(t_eval[0]), float(t_eval[-1])
    fwd = solve_ivp(_rhs, (t0, t1 + pad), init_cond, args=(alpha, beta),
                    method="RK45", rtol=rtol, atol=atol, dense_output=True)
    bwd = solve_ivp(_rhs, (t0, t0 - pad), init_cond, args=(alpha, beta),
                    method="RK45", rtol=rtol, atol=atol, dense_output=True)

    def prevalence(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t)
        late = t >= t0
        if late.any():
            out[late] = fwd.sol(t[late])[1]
        if (~late).any():
            out[~late] = bwd.sol(t[~late])[1]
        return out

    return prevalence


def warp(I_ode, t_eval, delta_t, a):
    """Shift and scale by interpolation -- matches the existing Fig 6/9 code."""
    f = interp1d(t_eval + delta_t, a * np.asarray(I_ode), kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return np.clip(f(t_eval), 0.0, None)


def warp_exact(prevalence, t_eval, delta_t, a):
    """Shift and scale by evaluating the ODE itself at t - delta_t."""
    return np.clip(a * prevalence(np.asarray(t_eval) - delta_t), 0.0, None)
