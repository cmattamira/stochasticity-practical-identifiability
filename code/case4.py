"""Coverage, case4 — the warped-ODE arm.

Step 3 of the coverage protocol, with synthetic data built by transforming the
deterministic solution rather than simulating:

    I_warped(t) = a * I_ODE(t - Delta_t)

Delta_t comes from the branching-process limit and is a function of the fitted
pair alone -- no calibration of any kind (time_shift.py). The scaling factor a
has to be measured, and comes either from the values extracted at the cell's
true parameters or from their mean and sd interpolated at the fitted pair's own
R0 (peak_scaling.py); --scaling-route picks which.

Why that distinction matters. The published generator drew both the shift and
the scaling factor by resampling values extracted at the TRUE parameters, and
reused them unchanged for every fitted pair, so the generator did not adapt to
the pair being bootstrapped from. That makes the calibration circular. Taking
the shift from theory removes it for that component outright; the parametric
scaling route reduces it to interpolation across the parameter grid for the
other.

No major-outbreak conditioning is applied here, and none is needed. A warped
ODE cannot fizzle -- it always produces an epidemic-shaped curve -- and the
shift is already conditioned by construction, since draw_time_shift conditions
the number of surviving branching lines on at least one.

The statistical code is unchanged: the ODE solve, the objective function
and the confidence-region estimator are copied verbatim from case3.py so the
arms stay comparable.

Usage:
    python case4.py --combo 7 --bootstrap 1000 --seed 12345 \
        --data /path/to/CoverageData --out /path/to/results \
        --scaling-route parametric --scaling-summary /path/to/summary_N1000.csv
"""
import argparse
import os
import numpy as np

_p = argparse.ArgumentParser()
_p.add_argument("--combo", type=int, required=True, help="parameter combination, 0-15")
_p.add_argument("--bootstrap", type=int, default=100, help="synthetic datasets per fitted pair")
_p.add_argument("--seed", type=int, required=True)
_p.add_argument("--data", required=True, help="directory holding params_combos.csv and best_fit_params/")
_p.add_argument("--out", required=True, help="directory to write this task's result CSV into")
_p.add_argument("--save-clouds", default=None,
                help="directory to write the bootstrap clouds into; omit to skip saving them")
_p.add_argument("--scaling-route", choices=("parametric", "empirical"),
                default="parametric",
                help="how to draw the peak scaling factor")
_p.add_argument("--scaling-summary", default=None,
                help="summary_N1000.csv, for --scaling-route parametric")
_p.add_argument("--scaling-samples", default=None,
                help="directory of samples_N1000_*.csv, for --scaling-route empirical")
_p.add_argument("--rho", type=float, default=0.0,
                help="correlation between the scaling factor and the shift; "
                     "0 draws them independently, which is what the published "
                     "coverage code did. Parametric route only.")
_args = _p.parse_args()

if _args.scaling_route == "parametric" and _args.scaling_summary is None:
    _p.error("--scaling-route parametric needs --scaling-summary")
if _args.scaling_route == "empirical" and _args.scaling_samples is None:
    _p.error("--scaling-route empirical needs --scaling-samples")

# The notebooks draw from numpy's legacy global generator. Seeding it here is
# what actually makes a run reproducible.
np.random.seed(_args.seed)

os.makedirs(_args.out, exist_ok=True)


def save_cloud(cloud, tag):
    """Write the bootstrap clouds to disk, if --save-clouds was given.

    The cloud is the (parameter, bootstrap replicate, fitted pair) array the
    coverage number is computed from. Keeping it means a different
    confidence-region estimator can be tried later without rerunning the fits.
    """
    if _args.save_clouds is None:
        return
    os.makedirs(_args.save_clouds, exist_ok=True)
    np.save(os.path.join(_args.save_clouds, tag + ".npy"), cloud)

import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from time_shift import draw_time_shift, SubcriticalParameters
from peak_scaling import (load_scaling_moments, load_peak_scaling_factors,
                          draw_scaling_factor)
from warp import ode_prevalence_extended, warp_exact

def dydt(t, y, par):

    par = np.abs(par)

    alpha = par[0]
    beta  = par[1]

    S = y[0]
    I = y[1]
    R = y[2]
    C = y[3]

    dS =  -beta*S*I
    dI = beta*S*I - alpha*I
    dR = alpha*I
    dC = beta*S*I

    return np.array([dS,dI,dR,dC])

# Solve the ODE model using solve_ivp

def SolveModel(par,tspan,init_cond,t_eval):

    ode_soln = solve_ivp(dydt,
                         tspan,
                         init_cond,
                         t_eval = t_eval,
                         args = (par,))
    return ode_soln

N0 = 1000
I0 = 5
R0 = 0
C0 = 0
S0 = N0 - I0 - R0
init_cond = np.array([S0, I0, R0, C0])

#Define time variables
t_final = 100
tspan = (1,t_final)
t_eval = np.linspace(1., t_final, t_final)
tdata = np.arange(0,t_final)

def objective_fun(par,tdata,ydatap,init_cond):

    tspan = (tdata[0],tdata[-1]);
    t_eval = tdata;

    soln = SolveModel(par,tspan,init_cond,t_eval)
    I = soln.y[1]

    SSR = np.sum((I-ydatap)**2)

    return SSR

def compute_2d_confidence_region(bootstrap_best_pars, true_pars, confidence_level):
    num_runs = bootstrap_best_pars.shape[2]
    true_params_in_contour = np.zeros(num_runs, dtype=bool)

    for i in range(num_runs):
        x = bootstrap_best_pars[0, :, i]
        y = bootstrap_best_pars[1, :, i]

        kde = gaussian_kde(np.vstack([x, y]))
        X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                           np.linspace(y.min(), y.max(), 100))
        Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])), X.shape)

        sorted_Z = np.sort(Z.ravel())[::-1]
        cumsum_Z = np.cumsum(sorted_Z) / np.sum(sorted_Z)
        level_idx = np.argmax(cumsum_Z >= confidence_level)
        level_val = sorted_Z[level_idx]

        true_density = kde(true_pars)
        true_params_in_contour[i] = true_density >= level_val

    return np.sum(true_params_in_contour) / num_runs




param_file = os.path.join(_args.data, "params_combos.csv")
params_df = pd.read_csv(param_file)
params_df["R0"] = (params_df["Beta"] * N0 / params_df["Alpha"]).round(2)

fit_param_dir = os.path.join(_args.data, "best_fit_params")

save_dir = _args.out
os.makedirs(save_dir, exist_ok=True)

method = 'Nelder-Mead'    # optimizer method

confidence_level = 0.68
n_bootstrap = _args.bootstrap

combo_index = _args.combo

if combo_index >= len(params_df):
    raise ValueError(f"Invalid combo index {combo_index} (max is {len(params_df)-1})")

row = params_df.iloc[combo_index]
alpha_true = row["Alpha"]
beta_true = row["Beta"]
R0 = round(beta_true * N0 / alpha_true, 2)
true_par = [alpha_true, beta_true]

fname = f"best_par_alpha{str(alpha_true).replace('.', 'p')}_beta{str(beta_true).replace('.', 'p')}.csv"
fit_path = os.path.join(fit_param_dir, fname)
if not os.path.exists(fit_path):
    raise FileNotFoundError(f"Missing file: {fname}")

best_pars = np.loadtxt(fit_path, delimiter=",")  # shape (2, 100)
n_runs = best_pars.shape[1]

# ---------- THE SCALING CALIBRATION ----------
# Loaded once, outside the loop. Under the parametric route this is a table of
# moments spanning the whole R0 grid, and each fitted pair reads its own values
# off it. Under the empirical route it is the list of scaling factors measured
# at THIS cell's true parameters, resampled for every fitted pair regardless of
# where that pair landed -- which is the circularity, left explicit.

if _args.scaling_route == "parametric":
    scaling_calibration = dict(
        moments_by_R0=load_scaling_moments(_args.scaling_summary))
else:
    scaling_calibration = dict(
        peak_scaling_factors=load_peak_scaling_factors(
            _args.scaling_samples, alpha_true, beta_true))


def warped_datasets(alpha, beta, n_datasets):
    """n_datasets synthetic prevalence series from one fitted pair.

    Raises SubcriticalParameters when the pair has R0 <= 1, where there is no
    epidemic to shift. The caller counts those rather than substituting a value.

    The ODE is solved once per fitted pair, not once per replicate. The window
    is padded to cover the largest shift actually drawn, so the solution is
    never read outside the interval it was integrated over, and no
    extrapolation is involved anywhere.
    """
    delta_t = draw_time_shift(alpha, beta, N0, I0, size=n_datasets)
    a = draw_scaling_factor(alpha, beta, N0, size=n_datasets,
                            rho=_args.rho, delta_t=delta_t,
                            **scaling_calibration)

    pad = float(np.abs(delta_t).max()) + 1.0
    prevalence = ode_prevalence_extended(alpha, beta, init_cond, t_eval, pad)

    data = np.empty((n_datasets, len(t_eval)))
    for m in range(n_datasets):
        data[m] = warp_exact(prevalence, t_eval, delta_t[m], a[m])
    return data


print(f"\n>>> Running coverage for combo #{combo_index}: "
      f"alpha = {alpha_true}, beta = {beta_true}, R0 = {R0}")
print(f"    scaling route: {_args.scaling_route}, rho = {_args.rho}")

clouds = []
n_subcritical = 0

for i in range(n_runs):
    best_fit_pars = np.abs(best_pars[:, i])

    try:
        simulated = warped_datasets(best_fit_pars[0], best_fit_pars[1], n_bootstrap)
    except SubcriticalParameters:
        # One of the hundred fitted pairs at R0 = 1.21 has R0_hat = 0.206. The
        # shift is undefined there, so the pair is dropped and counted. The
        # count goes in the result file.
        n_subcritical = n_subcritical + 1
        continue

    fits = np.zeros((np.size(true_par), n_bootstrap))
    for j in range(n_bootstrap):
        if j % 100 == 0:
            print(f"  -> Combo {combo_index}, Realization {i}, Bootstrap {j}")
        optimizer_bootstrap = minimize(
            objective_fun,
            best_fit_pars,
            args=(tdata, simulated[j], init_cond),
            method='Nelder-Mead'
        )
        fits[:, j] = np.abs(optimizer_bootstrap.x)

    clouds.append(fits)

if len(clouds) == 0:
    raise RuntimeError(f"every fitted pair was subcritical at combo {combo_index}")

# (parameters, bootstrap samples, fitted pairs), matching the other arms
bootstrap_best_pars = np.stack(clouds, axis=2)

save_cloud(bootstrap_best_pars,
           f"clouds_combo{combo_index}"
           f"_alpha{str(alpha_true).replace('.', 'p')}"
           f"_beta{str(beta_true).replace('.', 'p')}")

coverage = compute_2d_confidence_region(bootstrap_best_pars, true_par, confidence_level)

results = [{
    "Alpha_true": alpha_true,
    "Beta_true": beta_true,
    "R0": R0,
    "Confidence_Level": confidence_level,
    "Coverage": coverage,
    "Scaling_Route": _args.scaling_route,
    "Rho": _args.rho,
    "Pairs_Used": bootstrap_best_pars.shape[2],
    "Pairs_Subcritical": n_subcritical,
}]

# ---------- SAVE ----------
df = pd.DataFrame(results)
outname = f"coverage_combo{combo_index}_alpha{str(alpha_true).replace('.', 'p')}_beta{str(beta_true).replace('.', 'p')}.csv"
df.to_csv(os.path.join(save_dir, outname), index=False)
print(f"\ncoverage {coverage:.3f} from {bootstrap_best_pars.shape[2]} pairs "
      f"({n_subcritical} subcritical, dropped)")
