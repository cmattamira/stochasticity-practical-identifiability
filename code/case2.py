"""Coverage, case2 — run one parameter combination.

Derived from the notebook Coverage_case2_resampling_all_param_combos.ipynb.
The statistical code is unchanged from that notebook. Only paths, the combination index,
the bootstrap count and seeding were lifted out to command-line arguments.

Usage:
    python case2.py --combo 7 --bootstrap 100 --seed 12345 \
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
_p.add_argument("--save-clouds", default=None,
                help="directory to write the bootstrap clouds into; omit to skip saving them")
_args = _p.parse_args()

# The notebooks draw from numpy's legacy global generator. Seeding it here is
# what actually makes a run reproducible; the `rng = np.random.default_rng(42)`
# line further down is never used by the notebook code and is left in place
# untouched.
np.random.seed(_args.seed)

os.makedirs(_args.out, exist_ok=True)


def save_cloud(cloud, tag):
    """Write one bootstrap cloud to disk, if --save-clouds was given.

    The cloud is the (parameter, bootstrap replicate, fitted pair) array that
    the coverage number is computed from. Only the coverage was ever kept, so
    trying a different confidence-region estimator meant rerunning the fits --
    hours per cell. Keeping the cloud costs 1.6 MB and makes that free.

    Without the flag this does nothing, so a run that omits it is identical to
    one from before this function existed. No random numbers are drawn here.
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
from scipy.stats import gaussian_kde, skewnorm
from tqdm import tqdm


param_file = os.path.join(_args.data, "params_combos.csv")
fit_param_dir = os.path.join(_args.data, "best_fit_params")
ctmc_dir = os.path.join(_args.data, "ctmc_trajectories")
save_dir = _args.out
os.makedirs(save_dir, exist_ok=True)

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

def SIR_model(t, y, alpha, beta):
    S, I, R, C = y  # Include cumulative incidence C
    dSdt = -beta * S * I
    dIdt = beta * S * I - alpha * I
    dRdt = alpha * I
    dCdt = beta * S * I  # Cumulative incidence is driven by new infections
    return [dSdt, dIdt, dRdt, dCdt]

def SolveModel(par, tspan, y0, t_eval):
    alpha, beta = par
    soln = solve_ivp(SIR_model, tspan, y0, args=(alpha, beta), t_eval=t_eval, method='RK45')
    return soln

def objective_fun(par, tdata, ydata, init_cond):
    soln = SolveModel(par, (tdata[0], tdata[-1]), init_cond, tdata)
    I = soln.y[1]
    SSR = np.sum((I - ydata)**2)
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


# === Load Parameter Combinations ===
params_df = pd.read_csv(param_file)
params_df["R0"] = (params_df["Beta"] * N0 / params_df["Alpha"]).round(2)

# === Map (alpha, beta) to CTMC files ===
param_to_file = {
    (round(row["Alpha"], 3), round(row["Beta"], 4)): os.path.join(
        ctmc_dir,
        f"alpha{str(row['Alpha']).replace('.', 'p')}_beta{str(row['Beta']).replace('.', 'p')}.csv"
    )
    for _, row in params_df.iterrows()
}



# === Compute Residuals from CTMC Data ===
all_residuals = []

for idx, row in params_df.iterrows():
    alpha, beta, R0 = row["Alpha"], row["Beta"], row["R0"]
    par = (alpha, beta)
    file = param_to_file.get((round(alpha, 3), round(beta, 4)))

    if not os.path.exists(file):
        continue

    df = pd.read_csv(file)
    soln = SolveModel(par, tspan, init_cond, t_eval)
    I_ode = soln.y[1]
    I_peak = np.max(I_ode)
    dI_ode = np.gradient(I_ode, t_eval)
    t_peak = t_eval[np.argmax(I_ode)]

    for traj_id in df["trajectory"].unique():
        df_traj = df[df["trajectory"] == traj_id]
        df_traj = df_traj[df_traj["time"].isin(t_eval)].sort_values("time")

        for i, row_t in df_traj.iterrows():
            t = int(row_t["time"])
            t_idx = t - 1
            i_ctmc = row_t["I"]
            i_ode_val = I_ode[t_idx]

            if i_ode_val >= 0.001 * I_peak and i_ctmc > 0:
                residual = (i_ctmc - i_ode_val) / i_ode_val
                all_residuals.append({
                    "time": t,
                    "I_ODE": i_ode_val,
                    "dI_dt": dI_ode[t_idx],
                    "residual": residual,
                    "R0": R0,
                    "alpha": alpha,
                    "beta": beta,
                    "trajectory": traj_id,
                    "time_from_peak": t - t_peak,
                    "phase": "Before Peak" if t < t_peak else "After Peak"
                })

residuals_df = pd.DataFrame(all_residuals)

# Twenty bins, not a hundred (changed 11/9). A residual pool for one
# (alpha, beta, phase) is built from a single ODE trajectory sampled at 100
# integer days, and the rising limb of a fast epidemic supplies only a handful
# of distinct prevalence values -- at R0 = 11.4, about six. A hundred bins
# cannot be filled from six points, which is why most lookups used to find
# nothing and silently add no noise. At twenty, 99.3% of lookups land in a
# populated bin.
N_BINS = 20

residuals_df["I_bin"] = pd.cut(
    residuals_df["I_ODE"],
    bins=np.linspace(residuals_df["I_ODE"].min(), residuals_df["I_ODE"].max(),
                     N_BINS + 1),
    include_lowest=True
)

residual_lookup_param = {
    (row.alpha, row.beta, row.phase, row.I_bin): group["residual"].values
    for _, group in residuals_df.groupby(["alpha", "beta", "phase", "I_bin"])
    for row in group.head(1).itertuples()
}

# The categories of I_bin are the bins pd.cut made, in increasing order, so a
# category's position in that list is its slot. Going through the position
# avoids the interval labels, which pd.cut rounds to three significant figures
# and which therefore do not match the exact edges. (Corrected 11/9: the
# earlier version placed pools by kbin.left, which put 270 of 601 pools one
# slot too low and dropped 32 of them entirely.)
_bin_slot = {iv: j for j, iv in enumerate(residuals_df["I_bin"].cat.categories)}

_bin_edges = np.linspace(residuals_df["I_ODE"].min(),
                         residuals_df["I_ODE"].max(), N_BINS + 1)

# _build_pool_index added on 11/2, corrected 11/9
def _build_pool_index(a, b):
    """Residual pools for one (alpha, beta), with the list of populated bins."""
    idx = {"Before Peak": [None] * N_BINS, "After Peak": [None] * N_BINS}
    for (ka, kb, kphase, kbin), vals in residual_lookup_param.items():
        if ka == a and kb == b:
            j = _bin_slot.get(kbin)
            if j is not None:
                idx[kphase][j] = vals
    occupied = {ph: np.array([j for j in range(N_BINS) if idx[ph][j] is not None])
                for ph in idx}
    return idx, occupied

# _draw_eps added on 11/2, corrected 11/9
def _draw_eps(pool_index, phase, value):
    """One residual for this phase and prevalence.

    pd.cut makes bins closed on the right, so the bin for a value is the first
    whose upper edge is at or above it. If that bin holds no residuals we draw
    from the nearest populated bin in the same phase, rather than adding no
    noise at all. At twenty bins this fires for well under one point in a
    hundred, and the nearest bin is a neighbour.
    """
    idx, occupied = pool_index
    j = int(np.searchsorted(_bin_edges[1:], value, side="left"))
    j = min(max(j, 0), N_BINS - 1)
    pool = idx[phase][j]
    if pool is None or len(pool) == 0:
        if len(occupied[phase]) == 0:
            return 0.0
        k = occupied[phase][np.argmin(np.abs(occupied[phase] - j))]
        pool = idx[phase][k]
    return np.random.choice(pool)

# === Coverage Setup ===
confidence_level = 0.68
n_bootstrap = _args.bootstrap
combo_index = _args.combo

row = params_df.iloc[combo_index]
alpha_true = row["Alpha"]
beta_true = row["Beta"]
R0 = round(beta_true * N0 / alpha_true, 2)
true_par = [alpha_true, beta_true]

# === Load Best Fit Parameters ===
fname = f"best_par_alpha{str(alpha_true).replace('.', 'p')}_beta{str(beta_true).replace('.', 'p')}.csv"
fit_path = os.path.join(fit_param_dir, fname)
if not os.path.exists(fit_path):
    raise FileNotFoundError(f"Missing file: {fname}")

best_pars = np.loadtxt(fit_path, delimiter=",")
num_runs = best_pars.shape[1]

# changed 11/2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sample 100 synthetic trajectories per parameter combo and plot them
fig, axes = plt.subplots(4, 4, figsize=(20, 14))
axes = axes.flatten()

for idx, row in params_df.iterrows():
    alpha, beta, R0 = row["Alpha"], row["Beta"], row["R0"]
    par = (alpha, beta)
    ax = axes[idx]

    soln = SolveModel(par, tspan, init_cond, t_eval)
    I_ode = soln.y[1]
    t_peak = t_eval[np.argmax(I_ode)]

    # Generate 100 synthetic trajectories
    sims = []
    for _ in range(100):
        I_sim = []

        _pool_idx = _build_pool_index(alpha, beta) # added 11/2
        
        for i, t in enumerate(t_eval):
            I_val = I_ode[i]
            phase = "Before Peak" if t < t_peak else "After Peak"

            # Commented block replaced by new eps calculation on 11/2
            # matched_bin = None
            # for b in residual_lookup_param:
            #     bin_label = b[3]
            #     if b[0] == alpha and b[1] == beta and b[2] == phase and I_val in bin_label:
            #         matched_bin = b
            #         break

            # eps = np.random.choice(residual_lookup_param[matched_bin]) if matched_bin else 0.0
            
            eps = _draw_eps(_pool_idx, phase, I_val)
            I_sim.append(I_val * (1 + eps))

        sims.append(I_sim)

    # Plot the simulations
    for sim in sims:
        ax.plot(t_eval, sim)

    # Plot the ODE solution
    ax.plot(t_eval, I_ode, color='black', linewidth=2)
    ax.set_title(f"$\\alpha$={alpha}, $\\beta$={beta}, $R_0$={R0:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Infected")
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.suptitle("Synthetic I(t) from Empirical Residual Sampling (by α, β, Phase)", fontsize=20, y=1.02)
plt.show()


# === Main Bootstrap Loop Using Empirical Residuals ===
method = "Nelder-Mead"
bootstrap_best_pars = np.zeros((2, n_bootstrap, num_runs))

_pool_idx_true = _build_pool_index(alpha_true, beta_true) # added 11/2

for i in tqdm(range(num_runs), desc=f"Combo {combo_index}"):
    fit_par = best_pars[:, i]
    sol_clean = SolveModel(fit_par, tspan, init_cond, t_eval)
    I_clean = sol_clean.y[1]
    t_peak = t_eval[np.argmax(I_clean)]

    simulated_noisy_trajectories = np.zeros((len(t_eval), n_bootstrap))

    for j in range(n_bootstrap):
        for t_idx, t in enumerate(t_eval):
            base_val = I_clean[t_idx]
            phase = "Before Peak" if t < t_peak else "After Peak"

            # matched_bin = None
            # for b in residual_lookup_param:
            #     bin_label = b[3]
            #     if b[0] == alpha_true and b[1] == beta_true and b[2] == phase and base_val in bin_label:
            #         matched_bin = b
            #         break

            # eps = np.random.choice(residual_lookup_param[matched_bin]) if matched_bin else 0.0
            # noisy_val = base_val * (1 + eps)

            eps = _draw_eps(_pool_idx_true, phase, base_val)
            
            simulated_noisy_trajectories[t_idx, j] = base_val * (1 + eps) # uptdate 11/2


    for j in range(n_bootstrap):
        noisy_I = simulated_noisy_trajectories[:, j]
        res = minimize(objective_fun, fit_par, args=(t_eval, noisy_I, init_cond), method=method)
        bootstrap_best_pars[:, j, i] = np.abs(res.x)


# === Compute and Save Coverage ===
save_cloud(bootstrap_best_pars,
           f"clouds_combo{combo_index}"
           f"_alpha{str(alpha_true).replace('.', 'p')}"
           f"_beta{str(beta_true).replace('.', 'p')}")

coverage = compute_2d_confidence_region(bootstrap_best_pars, true_par, confidence_level)

df = pd.DataFrame([{
    "Alpha_true": alpha_true,
    "Beta_true": beta_true,
    "R0": R0,
    "Confidence_Level": confidence_level,
    "Coverage": coverage
}])

outname = f"coverage_combo{combo_index}_alpha{str(alpha_true).replace('.', 'p')}_beta{str(beta_true).replace('.', 'p')}.csv"
df.to_csv(os.path.join(save_dir, outname), index=False)
