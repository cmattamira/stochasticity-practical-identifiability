"""Empirical residual noise: the lookup, and drawing from it.

Extracted from case2.py on 22 November so that the figure scripts and case2.py
can share one copy instead of keeping two that drift apart. The function bodies
are case2.py's, unchanged. The only difference is that what used to be module
level globals in case2.py -- the residual table, the bin slots and the bin
edges -- are now carried in a dictionary and passed in, so the module holds no
state of its own and can serve several parameter sets in one session.

How the method works. For each parameter combination we take the saved CTMC
trajectories, compute the relative residual (I_ctmc - I_ode)/I_ode at each
observation time, and file it by parameter pair, epidemic phase (before or
after the deterministic peak) and the prevalence of the ODE at that time.
Synthetic data are then built by drawing a residual from the matching pool and
applying it multiplicatively. The point of the method is that the noise is
empirical in its magnitude and phase dependence, while remaining independent
between time points -- which is what the paper then shows is not enough.

Twenty bins, not a hundred. A residual pool for one (alpha, beta, phase) is
built from a single ODE trajectory sampled at 100 integer days, and the rising
limb of a fast epidemic supplies only a handful of distinct prevalence values
-- at R0 = 11.4, about six. A hundred bins cannot be filled from six points,
which is why most lookups used to find nothing and silently add no noise. At
twenty, 99.3% of lookups land in a populated bin.

    import empirical_noise as en
    residuals = en.build_residuals(params_df, ctmc_dir, prevalence_of, t_eval)
    lookup    = en.build_lookup(residuals)
    pools     = en.pool_index(lookup, alpha, beta)
    eps       = en.draw_residual(lookup, pools, "Before Peak", I_value)
"""

import os

import numpy as np
import pandas as pd

N_BINS = 20


def build_residuals(params_df, ctmc_dir, prevalence_of, t_eval, N0=1000):
    """Relative residuals from the saved CTMC trajectories, for every cell.

    prevalence_of(par, t_eval) must return the ODE prevalence on t_eval; it is
    passed in rather than imported so this module needs no ODE solver of its
    own and cannot disagree with the caller's.

    A point is kept only where the ODE prevalence is at least 0.1% of its peak
    and the stochastic trajectory is non-zero, which keeps the ratio away from
    division by something indistinguishable from zero.
    """
    params_df = params_df.copy()
    params_df["R0"] = (params_df["Beta"] * N0 / params_df["Alpha"]).round(2)

    param_to_file = {
        (round(row["Alpha"], 3), round(row["Beta"], 4)): os.path.join(
            ctmc_dir,
            f"alpha{str(row['Alpha']).replace('.', 'p')}"
            f"_beta{str(row['Beta']).replace('.', 'p')}.csv")
        for _, row in params_df.iterrows()
    }

    all_residuals = []
    for _, row in params_df.iterrows():
        alpha, beta, R0 = row["Alpha"], row["Beta"], row["R0"]
        file = param_to_file.get((round(alpha, 3), round(beta, 4)))
        if file is None or not os.path.exists(file):
            continue

        df = pd.read_csv(file)
        I_ode = prevalence_of((alpha, beta), t_eval)
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

    return pd.DataFrame(all_residuals)


def build_lookup(residuals_df, n_bins=N_BINS):
    """Bin the residuals and index them by (alpha, beta, phase, bin).

    Returns everything the drawing functions need, so no state is left in this
    module. The bin slots are taken from the positions of the categories pd.cut
    produced, not from their interval labels: pd.cut rounds those labels to
    three significant figures, so they do not match the exact edges.
    """
    residuals_df = residuals_df.copy()
    edges = np.linspace(residuals_df["I_ODE"].min(),
                        residuals_df["I_ODE"].max(), n_bins + 1)
    residuals_df["I_bin"] = pd.cut(residuals_df["I_ODE"], bins=edges,
                                   include_lowest=True)

    pools = {
        (row.alpha, row.beta, row.phase, row.I_bin): group["residual"].values
        for _, group in residuals_df.groupby(["alpha", "beta", "phase", "I_bin"])
        for row in group.head(1).itertuples()
    }

    bin_slot = {iv: j for j, iv
                in enumerate(residuals_df["I_bin"].cat.categories)}

    return dict(pools=pools, bin_slot=bin_slot, bin_edges=edges, n_bins=n_bins)


def pool_index(lookup, alpha, beta):
    """Residual pools for one (alpha, beta), with the list of populated bins."""
    n_bins = lookup["n_bins"]
    idx = {"Before Peak": [None] * n_bins, "After Peak": [None] * n_bins}
    for (ka, kb, kphase, kbin), vals in lookup["pools"].items():
        if ka == alpha and kb == beta:
            j = lookup["bin_slot"].get(kbin)
            if j is not None:
                idx[kphase][j] = vals
    occupied = {ph: np.array([j for j in range(n_bins) if idx[ph][j] is not None])
                for ph in idx}
    return idx, occupied


def draw_residual(lookup, pools, phase, value):
    """One residual for this phase and prevalence.

    pd.cut makes bins closed on the right, so the bin for a value is the first
    whose upper edge is at or above it. If that bin holds no residuals we draw
    from the nearest populated bin in the same phase, rather than adding no
    noise at all. At twenty bins this fires for well under one point in a
    hundred, and the nearest bin is a neighbour.

    Draws come from numpy's legacy global generator, as the rest of the
    coverage code does.
    """
    n_bins = lookup["n_bins"]
    idx, occupied = pools
    j = int(np.searchsorted(lookup["bin_edges"][1:], value, side="left"))
    j = min(max(j, 0), n_bins - 1)
    pool = idx[phase][j]
    if pool is None or len(pool) == 0:
        if len(occupied[phase]) == 0:
            return 0.0
        k = occupied[phase][np.argmin(np.abs(occupied[phase] - j))]
        pool = idx[phase][k]
    return np.random.choice(pool)
