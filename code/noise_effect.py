"""How much of the hybrid method's spread comes from the undocumented noise?

Reimplements get_warped_data from Coverage_case4.ipynb exactly as published,
then the same generator with the AR(1) noise term removed, and compares the
coefficient of variation of the fitted parameters. The noise term appears in
neither the manuscript's equation nor its text.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d

N0, I0 = 1000, 5
t_eval = np.linspace(1., 100., 100)
init_cond = np.array([N0 - I0, I0, 0, 0])


def rhs(t, y, a, b):
    S, I, R, C = y
    return [-b*S*I, b*S*I - a*I, a*I, b*S*I]


def ode_I(par):
    a, b = np.abs(par)
    s = solve_ivp(rhs, (1., 100.), init_cond, args=(a, b), t_eval=t_eval,
                  rtol=1e-6, atol=1e-8)
    return s.y[1]


def objective_fun(par, ydata):
    return np.sum((ode_I(par) - ydata) ** 2)


def get_warped_data(par, peak_shifts, peak_scaling_factors, with_noise):
    """Cell 6 of Coverage_case4.ipynb, with the noise term switchable."""
    I = ode_I(par)
    T = len(I)
    time_grid = np.arange(T)

    shift_sample = np.random.choice(peak_shifts)
    scale_sample = np.random.choice(peak_scaling_factors)

    time_warp = np.clip(time_grid - shift_sample, 0, T - 1)
    interp_func = interp1d(time_grid, I, kind="cubic", fill_value="extrapolate")
    warped_curve = interp_func(time_warp)

    noise = np.zeros(T)
    if with_noise:
        rho = 0.9
        sigma_noise = np.random.uniform(2.0, 4.0)
        for t in range(1, T):
            noise[t] = rho * noise[t-1] + np.random.normal(0, sigma_noise)

    return np.clip(scale_sample * (warped_curve + noise), 0.0, None)


def cv_of_fits(true_par, peak_shifts, peak_scaling, with_noise, M, seed):
    np.random.seed(seed)
    fits = np.zeros((2, M))
    for m in range(M):
        data = get_warped_data(true_par, peak_shifts, peak_scaling, with_noise)
        res = minimize(objective_fun, true_par, args=(data,), method="Nelder-Mead")
        fits[:, m] = np.abs(res.x)
    return (100 * fits.std(axis=1, ddof=1) / fits.mean(axis=1))


if __name__ == "__main__":
    import glob
    M = 300
    for tag, alpha, beta in (("R0 = 4.00", 0.10, 0.0004),
                             ("R0 = 1.21", 0.33, 0.0004)):
        f = ("../gen3/samples/samples_N1000_alpha%s_beta%s.npz"
             % (str(alpha).replace('.', 'p'), str(beta).replace('.', 'p')))
        d = np.load(f)
        major = d["major"]
        shifts = d["dt_grid"][major]
        scales = d["a_grid"][major]
        print(f"\n{tag}   (alpha = {alpha}, beta = {beta}),  M = {M}")
        print(f"  empirical shift sd {shifts.std(ddof=1):.2f} d, "
              f"scale sd {scales.std(ddof=1):.3f}, n = {major.sum()}")
        for with_noise, label in ((True, "as published (with AR(1) noise)"),
                                  (False, "noise term removed")):
            cva, cvb = cv_of_fits([alpha, beta], shifts, scales, with_noise, M, 20261116)
            print(f"  {label:34s}  CV(alpha) {cva:6.2f}   CV(beta) {cvb:6.2f}")
