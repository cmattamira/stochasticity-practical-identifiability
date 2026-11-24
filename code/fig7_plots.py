# Two figures from the regenerated summaries:
#   Fig7_conditioning.png -- the original 2x2 layout, conditioned against
#       unconditioned, over the full R0 range, with the published figure's
#       R0 >= 2 cut marked rather than applied.
#   Fig7_upgrade.png      -- the proposed replacement: theory overlaid on the
#       shift scale, the amplitude collapsed by sqrt(N), and the two components
#       in common units.
# Colours are the paper's; each N also carries its own marker, because the
# existing orange/green pair is indistinguishable under protanopia.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from time_shift import shift_moments

# Summaries are read from data/fig7 and figures are written to figures/,
# both relative to the repository root.
SUM = os.path.join(HERE, "..", "data", "fig7")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
I0 = 5
STYLE = {100: ("#1f77b4", "o"), 1000: ("#ff7f0e", "s"), 10000: ("#2ca02c", "^")}

d = {N: pd.read_csv(os.path.join(SUM, "summary_N%d.csv" % N)).sort_values("R0")
     for N in STYLE}

# ---------------------------------------------------------------- figure one
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
panels = [("mean_dt", "Mean Peak Time Shift ($\\Delta t$)", "Mean($\\Delta t$)", False),
          ("mean_a",  "Mean Amplitude Scale ($a$)",          "Mean($a$)",        False),
          ("std_dt",  "Std of Peak Time Shift ($\\Delta t$)", "Std($\\Delta t$)", True),
          ("std_a",   "Std of Amplitude Scale ($a$)",         "Std($a$)",         True)]

for ax, (stat, title, ylab, logy) in zip(axs.ravel(), panels):
    for N, (c, m) in STYLE.items():
        s = d[N]
        ax.plot(s.R0, s[stat + "_all_exact"], m + "--", color=c, ms=6,
                mfc="none", mew=1.3, lw=1.0, alpha=0.65,
                label="N = %d, all" % N)
        ax.plot(s.R0, s[stat + "_major_exact"], m + "-", color=c, ms=6,
                lw=1.6, label="N = %d, major only" % N)
    ax.axvline(2.0, color="0.45", ls=":", lw=1.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$R_0$"); ax.set_ylabel(ylab)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axs[0, 0].annotate("published figure\nstarts here", xy=(2.0, 0.0),
                   xytext=(3.4, -9), fontsize=8, color="0.35",
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
axs[0, 0].legend(frameon=False, fontsize=7.5, ncol=2)
fig.suptitle("Effect of major-outbreak conditioning (exact peak extraction)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "Fig7_conditioning.png"), dpi=160)
fig.savefig(os.path.join(OUT, "Fig7_conditioning.pdf"))

# ---------------------------------------------------------------- figure two
rows = []
for N in STYLE:
    s = d[N].copy()
    s["r"] = s.beta_scaled * s.N - s.alpha
    s["r_sd_dt"] = s.r * s.std_dt_major_exact
    s["c_R0"] = s.std_a_major_exact * np.sqrt(s.N)
    s["theory"] = [shift_moments(a, b, 1000, I0)[1] * (a * (R - 1.0))
                   for a, b, R in zip(s.alpha, s.alpha * s.R0 / 1000.0, s.R0)]
    rows.append(s)
allr = pd.concat(rows)

R0c = np.linspace(1.15, 15.0, 300)
th_c = np.array([shift_moments(0.10, 0.10 * R / 1000.0, 1000, I0)[1]
                 * (0.10 * (R - 1.0)) for R in R0c])

fig2, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for N, (c, m) in STYLE.items():
    s = allr[allr.N == N]
    ax[0].plot(s.R0, s.r_sd_dt, m, color=c, ms=7, mfc="none", mew=1.6,
               label="N = %d" % N)
ax[0].plot(R0c, th_c, "k-", lw=2, label="theory, $\\sqrt{\\mathrm{Var}\\log W}$")
ax[0].axvline(2.0, color="0.45", ls=":", lw=1.3)
ax[0].set_xlabel("$R_0$"); ax[0].set_ylabel("$r\\,\\mathrm{sd}(\\Delta t)$")
ax[0].set_title("A. Shift scale, $\\alpha$ removed"); ax[0].set_ylim(0, 2.2)
ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=0.25, lw=0.6)

for N, (c, m) in STYLE.items():
    s = allr[allr.N == N]
    ax[1].plot(s.R0, s.c_R0, m, color=c, ms=7, mfc="none", mew=1.6,
               label="N = %d" % N)
ax[1].axvline(2.0, color="0.45", ls=":", lw=1.3)
ax[1].set_xlabel("$R_0$"); ax[1].set_ylabel("$\\mathrm{sd}(a)\\sqrt{N} = c(R_0)$")
ax[1].set_title("B. Amplitude scale, collapsed by $\\sqrt{N}$")
ax[1].set_yscale("log"); ax[1].legend(frameon=False, fontsize=9)
ax[1].grid(alpha=0.25, lw=0.6, which="both")

ax[2].plot(R0c, th_c, "k-", lw=2, label="shift")
for N, (c, m) in STYLE.items():
    s = allr[allr.N == N]
    ax[2].plot(s.R0, s.std_a_major_exact, m + "--", color=c, ms=7, mfc="none",
               mew=1.6, lw=1.1, label="amplitude, N = %d" % N)
ax[2].axvline(2.0, color="0.45", ls=":", lw=1.3)
ax[2].set_xlabel("$R_0$"); ax[2].set_ylabel("fractional perturbation of $\\log I$")
ax[2].set_title("C. Both components, same units"); ax[2].set_yscale("log")
ax[2].legend(frameon=False, fontsize=8.5); ax[2].grid(alpha=0.25, lw=0.6, which="both")

fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "Fig7_upgrade.png"), dpi=160)
fig2.savefig(os.path.join(OUT, "Fig7_upgrade.pdf"))
print("written both figures")

print("\nPanel A, empirical / theory, major outbreaks, exact peaks:")
for N in STYLE:
    s = allr[(allr.N == N) & (allr.R0 >= 2)]
    print("   N = %-6d R0>=2: median %.2f  range %.2f-%.2f"
          % (N, np.median(s.r_sd_dt/s.theory), (s.r_sd_dt/s.theory).min(),
             (s.r_sd_dt/s.theory).max()))
    s1 = allr[(allr.N == N) & (allr.R0 < 2)]
    print("                R0=1.21: %.2f" % float((s1.r_sd_dt/s1.theory).iloc[0]))
