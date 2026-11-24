"""Does empirical_noise.py behave exactly like the code it was taken from?

case2.py produced the published empirical-noise numbers, so the extraction is
only safe if it is a pure move. This rebuilds the lookup both ways on the same
data -- once with case2.py's original module-level code, pasted in below
unchanged, and once through the module -- and checks that:

  1. the residual tables are identical, row for row;
  2. the pools are identical, key for key and value for value;
  3. drawing from the same seed gives the same residuals.

Anything less than all three and case2.py should not be switched over.

    python verify_extraction.py --data <dir with params_combos.csv and ctmc/>
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import empirical_noise as en

N0, I0 = 1000, 5
init_cond = np.array([N0 - I0, I0, 0, 0])
t_final = 100
tspan = (1, t_final)
t_eval = np.linspace(1., t_final, t_final)


def dydt(t, y, par):
    par = np.abs(par)
    alpha, beta = par[0], par[1]
    S, I, R, C = y
    return np.array([-beta*S*I, beta*S*I - alpha*I, alpha*I, beta*S*I])


def SolveModel(par, tspan, init_cond, t_eval):
    return solve_ivp(dydt, tspan, init_cond, t_eval=t_eval, args=(par,))


def prevalence_of(par, times):
    return SolveModel(par, (times[0], times[-1]), init_cond, times).y[1]


# =====================================================================
# case2.py's original code, pasted unchanged except that the paths come
# from arguments. This is the thing being compared against.
# =====================================================================
def original(params_df, ctmc_dir):
    params_df = params_df.copy()
    params_df["R0"] = (params_df["Beta"] * N0 / params_df["Alpha"]).round(2)

    param_to_file = {
        (round(row["Alpha"], 3), round(row["Beta"], 4)): os.path.join(
            ctmc_dir,
            f"alpha{str(row['Alpha']).replace('.', 'p')}_beta{str(row['Beta']).replace('.', 'p')}.csv"
        )
        for _, row in params_df.iterrows()
    }

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
                        "time": t, "I_ODE": i_ode_val, "dI_dt": dI_ode[t_idx],
                        "residual": residual, "R0": R0, "alpha": alpha,
                        "beta": beta, "trajectory": traj_id,
                        "time_from_peak": t - t_peak,
                        "phase": "Before Peak" if t < t_peak else "After Peak"})

    residuals_df = pd.DataFrame(all_residuals)

    N_BINS = 20
    residuals_df["I_bin"] = pd.cut(
        residuals_df["I_ODE"],
        bins=np.linspace(residuals_df["I_ODE"].min(),
                         residuals_df["I_ODE"].max(), N_BINS + 1),
        include_lowest=True)

    residual_lookup_param = {
        (row.alpha, row.beta, row.phase, row.I_bin): group["residual"].values
        for _, group in residuals_df.groupby(["alpha", "beta", "phase", "I_bin"])
        for row in group.head(1).itertuples()}

    _bin_slot = {iv: j for j, iv in enumerate(residuals_df["I_bin"].cat.categories)}
    _bin_edges = np.linspace(residuals_df["I_ODE"].min(),
                             residuals_df["I_ODE"].max(), N_BINS + 1)

    def _build_pool_index(a, b):
        idx = {"Before Peak": [None] * N_BINS, "After Peak": [None] * N_BINS}
        for (ka, kb, kphase, kbin), vals in residual_lookup_param.items():
            if ka == a and kb == b:
                j = _bin_slot.get(kbin)
                if j is not None:
                    idx[kphase][j] = vals
        occupied = {ph: np.array([j for j in range(N_BINS) if idx[ph][j] is not None])
                    for ph in idx}
        return idx, occupied

    def _draw_eps(pool_index, phase, value):
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

    return residuals_df, residual_lookup_param, _build_pool_index, _draw_eps


# =====================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--ctmc", required=True)
    args = p.parse_args()

    params_df = pd.read_csv(args.params)

    print("building both ways ...")
    res_old, pools_old, build_old, draw_old = original(params_df, args.ctmc)
    res_new = en.build_residuals(params_df, args.ctmc, prevalence_of, t_eval, N0)
    lookup_new = en.build_lookup(res_new)

    ok = True

    # 1. the residual tables.
    # original() returns the table after the I_bin column has been added;
    # build_residuals returns it before, because the binning belongs to
    # build_lookup. So the comparison is over the ten columns both produce.
    shared = list(res_new.columns)
    same_rows = len(res_old) == len(res_new)
    same_vals = same_rows and all(
        (np.array_equal(res_old[c].to_numpy(), res_new[c].to_numpy())
         if res_old[c].dtype.kind in "fi" else bool((res_old[c] == res_new[c]).all()))
        for c in shared)
    print(f"1. residual tables identical          : {same_vals}"
          f"   ({len(res_old)} rows, {len(shared)} columns)")
    ok &= bool(same_vals)

    # 2. the pools
    keys_old, keys_new = set(pools_old), set(lookup_new["pools"])
    same_keys = keys_old == keys_new
    same_pools = same_keys and all(
        np.array_equal(pools_old[k], lookup_new["pools"][k]) for k in keys_old)
    print(f"2. pools identical                    : {same_pools}"
          f"   ({len(keys_old)} pools)")
    ok &= bool(same_pools)

    # 3. the draws, from the same seed
    draws_match = True
    for _, row in params_df.iterrows():
        a, b = row["Alpha"], row["Beta"]
        pi_old = build_old(a, b)
        pi_new = en.pool_index(lookup_new, a, b)
        I_ode = prevalence_of((a, b), t_eval)
        t_peak = t_eval[np.argmax(I_ode)]
        for t_idx, value in enumerate(I_ode):
            phase = "Before Peak" if t_eval[t_idx] < t_peak else "After Peak"
            np.random.seed(1000 + t_idx)
            e_old = draw_old(pi_old, phase, value)
            np.random.seed(1000 + t_idx)
            e_new = en.draw_residual(lookup_new, pi_new, phase, value)
            if not np.isclose(e_old, e_new, rtol=0, atol=0):
                draws_match = False
                print(f"   MISMATCH at alpha={a} beta={b} t_idx={t_idx}: "
                      f"{e_old} vs {e_new}")
                break
        if not draws_match:
            break
    print(f"3. draws identical from the same seed : {draws_match}"
          f"   ({len(params_df)} cells x {len(t_eval)} times)")
    ok &= bool(draws_match)

    print()
    print("EXTRACTION IS A PURE MOVE" if ok else "DIFFERENCES FOUND -- do not switch case2.py over")
