"""Coverage, case3 — run one parameter combination.

Derived from the notebook Coverage_case3_all_param_combos.ipynb.
The statistical code is unchanged from that notebook. Only paths, the combination index,
the bootstrap count and seeding were lifted out to command-line arguments.

Usage:
    python case3.py --combo 7 --bootstrap 100 --seed 12345 \
        --data /path/to/CoverageData --out /path/to/results
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
_args = _p.parse_args()

# The notebooks draw from numpy's legacy global generator. Seeding it here is
# what actually makes a run reproducible; the `rng = np.random.default_rng(42)`
# line further down is never used by the notebook code and is left in place
# untouched.
np.random.seed(_args.seed)

os.makedirs(_args.out, exist_ok=True)

import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from tqdm import tqdm

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

def sim_SIR_CTMC(par):

    t0 = 1.0
    alpha = par[0]
    beta = par[1]
    maxsim = 20000000
    init_cond = np.array([S0, I0, R0, C0])
    x = []
    x = np.array([init_cond])

    t = np.array([t0])
    tmax = t_final

    report_time = 2
    t_report = [1]
    x_report = x


    # The possible events are a new infection or a recovery.
    # If a new infection occurs, subtract 1 from S, add 1 to I, add 1 to C.
    # If a recovery occurs, subtract 1 from I, and 1 to R
    transition_matrix = np.array([[-1,1,0,1],[0,-1,1,0]])

    for i in range(1,maxsim):

        if i == maxsim-1:
            print('Warning: maxsim reached')

        S,I,R,C = x[-1]

        transition_rates = np.array([beta*S*I,alpha*I])
        sum_rates = np.sum(transition_rates)

        if sum_rates>0:

            # Compute transition probabilities
            transition_probabilities = transition_rates/sum_rates

            # Choose time of next event and append to previous times t
            interevent_time = np.random.exponential(1/sum_rates)
            t = np.append(t, t[-1] + interevent_time)

#             print(t[-1],report_time)

            if t[-1]>report_time:
                t_report = np.append(t_report,report_time)#np.floor(t[-1]))
                x_report = np.append(x_report,[x[-1]],0)
                report_time = report_time + 1 #t_report[-1]

            # Choose which event happens.  This is done by finding smallest index such that a uniform random variable u
            # is smaller than the cumulative sum of the transition probabilties. Append current state to previous states x.
            u = np.random.uniform()
            cum_probs = np.cumsum(transition_probabilities)
            which_event = np.argmax(cum_probs>u)
            x_current = np.array(x[-1] + transition_matrix[which_event,:])

            x = np.append(x,[x_current],0)

        else:

            t_report = np.append(t_report,report_time)
            x_report = np.append(x_report,[x_report[-1]],0)
            report_time = report_time+1


        if report_time > tmax:
                break


        #if t[-1]>tmax:
         #   break


    return t_report, x_report

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
rng = np.random.default_rng(42)

confidence_level = 0.68
n_bootstrap = _args.bootstrap

combo_index = _args.combo

params_df = pd.read_csv(param_file)
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
num_runs = best_pars.shape[1]

results = []
print(f"\n>>> Running coverage for combo #{combo_index}: α = {alpha_true}, β = {beta_true}")

n_runs = best_pars.shape[1]
bootstrap_best_pars = np.zeros((np.size(true_par), n_bootstrap, n_runs))  # (parameters, bootstrap samples, CTMC realizations)

for i in range(n_runs):
    best_fit_pars = best_pars[:, i]

    for j in range(n_bootstrap):
        if j % 10 == 0:
            print(f"  → Combo {combo_index}, Realization {i}, Bootstrap {j}")

        [t, simulated_data] = sim_SIR_CTMC(best_fit_pars)
        optimizer_bootstrap = minimize(
            objective_fun,
            best_fit_pars,
            args=(tdata, simulated_data[:, 1], init_cond),
            method='Nelder-Mead'
        )
        bootstrap_best_pars[:, j, i] = np.abs(optimizer_bootstrap.x)

coverage = compute_2d_confidence_region(bootstrap_best_pars, true_par, confidence_level)

results.append({
    "Alpha_true": alpha_true,
    "Beta_true": beta_true,
    "R0": R0,
    "Confidence_Level": confidence_level,
    "Coverage": coverage
})

# ---------- SAVE ----------
df = pd.DataFrame(results)
outname = f"coverage_combo{combo_index}_alpha{str(alpha_true).replace('.', 'p')}_beta{str(beta_true).replace('.', 'p')}.csv"
df.to_csv(os.path.join(save_dir, outname), index=False)

