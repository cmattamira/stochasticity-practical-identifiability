"""How noisy are these CVs, and how much does the shift-scale correlation move them?

Two questions, because the first has to be answered before the second means
anything. Repeating one condition at M = 1000 gave CV(alpha) of 6.35 and 6.93
on two different seeds -- 9% apart, where the textbook Monte Carlo error on a
CV from 1000 draws is about 2%. The fitted parameters are heavy-tailed, so the
standard deviation in the numerator is dominated by a few extreme fits and is
much less stable than the usual formula assumes.

Part 1 measures that directly: the same condition on several independent seeds.
Part 2 compares rho = 0, -0.09 and -0.22 on the SAME drawn shifts, so the noise
that part 1 measures largely cancels and a small effect is still visible.

    python rho_sensitivity.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from validate_case4 import warped, fit_all, cv, N0, I0
from time_shift import draw_time_shift, shift_moments
from peak_scaling import load_scaling_moments, scaling_moments

# Scaling-factor summary written by fig7_regen.py, relative to this file.
import os
SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "fig7", "summary_N1000.csv")

alpha, beta, M = 0.10, 0.0004, 1000
moments = load_scaling_moments(SUMMARY)
mean_a, sd_a = scaling_moments(alpha, beta, N0, moments)
mean_dt, sd_dt = shift_moments(alpha, beta, N0, I0)

print(f"alpha = {alpha}, beta = {beta}, M = {M}")
print(f"shift  mean {mean_dt:+.3f}  sd {sd_dt:.3f}")
print(f"scale  mean {mean_a:.4f}  sd {sd_a:.4f}\n")

# ---------------------------------------------------------------- part 1
print("Part 1 -- the same condition on four independent seeds")
print("(theory shift, parametric scale, drawn independently)\n")
vals = []
for seed in (11, 22, 33, 44):
    np.random.seed(seed)
    dt = draw_time_shift(alpha, beta, N0, I0, size=M)
    a = mean_a + sd_a * np.random.normal(size=M)
    c = cv(fit_all(warped(alpha, beta, dt, a), [alpha, beta]))
    vals.append(c)
    print(f"  seed {seed:3d}   CV(alpha) {c[0]:6.2f}   CV(beta) {c[1]:6.2f}")
vals = np.array(vals)
print(f"\n  spread over seeds: CV(alpha) {vals[:,0].mean():.2f} "
      f"+/- {vals[:,0].std(ddof=1):.2f}   "
      f"CV(beta) {vals[:,1].mean():.2f} +/- {vals[:,1].std(ddof=1):.2f}")
print(f"  the textbook error would be about {100/np.sqrt(2*M):.1f}% of each value\n")

# ---------------------------------------------------------------- part 2
print("Part 2 -- rho on the SAME shifts, so the above noise cancels\n")
np.random.seed(707)
dt = draw_time_shift(alpha, beta, N0, I0, size=M)
z_dt = (dt - mean_dt) / sd_dt
z_indep = np.random.normal(size=M)          # the same normal for every rho

for rho in (0.0, -0.09, -0.22):
    z = rho * z_dt + np.sqrt(1 - rho ** 2) * z_indep
    a = mean_a + sd_a * z
    c = cv(fit_all(warped(alpha, beta, dt, a), [alpha, beta]))
    print(f"  rho {rho:+.2f}  (realised {np.corrcoef(a, dt)[0,1]:+.3f})   "
          f"CV(alpha) {c[0]:6.2f}   CV(beta) {c[1]:6.2f}")

print("\nCTMC benchmark at M = 1000:  CV(alpha) 4.95   CV(beta) 9.75")
