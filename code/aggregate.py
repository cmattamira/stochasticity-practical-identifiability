"""Gather the per-task result CSVs into one coverage table.

    python aggregate.py ~/identifiability/results coverage_table.csv

Each task writes one file per (case, combination); case 1 writes one row per
noise level. This stacks them, pivots to the layout of the manuscript's
coverage table, and reports anything missing.
"""
import pathlib
import sys

import pandas as pd

results = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("coverage_table.csv")

LABEL = {"case1": "Gaussian", "case2": "Empirical", "case3": "CTMC"}

rows = []
for case_dir in sorted(results.iterdir()):
    if not case_dir.is_dir():
        continue
    files = sorted(case_dir.glob("coverage_combo*.csv"))
    print(f"{case_dir.name}: {len(files)}/16 combinations present")
    missing = set(range(16)) - {int(f.name.split("combo")[1].split("_")[0]) for f in files}
    if missing:
        print(f"   missing combos: {sorted(missing)}")
    for f in files:
        df = pd.read_csv(f)
        df["Case"] = case_dir.name
        df["Method"] = LABEL.get(case_dir.name, case_dir.name)
        rows.append(df)

if not rows:
    sys.exit("no result files found")

allr = pd.concat(rows, ignore_index=True)

# Recompute R0 from alpha and beta rather than trusting the stored column.
# Case 2 writes the wrong value there: in that notebook the figure cell runs
# after the coverage setup and rebinds R0 as its loop variable, so every case 2
# file carries the R0 of the *last* parameter combination (2.42). Alpha_true and
# Beta_true are correct, so this recovers it. Without this the pivot below would
# put case 2 on its own index rows and fill the rest with NaN.
N0 = 1000
allr["R0"] = (allr["Beta_true"] * N0 / allr["Alpha_true"]).round(2)

# case 1 carries a Sigma column; label each noise level separately
if "Sigma" in allr.columns:
    m = allr["Sigma"].notna()
    allr.loc[m, "Method"] = allr.loc[m].apply(
        lambda r: f"Gaussian (sigma={r['Sigma']:g})", axis=1)

table = (allr.pivot_table(index=["Alpha_true", "Beta_true", "R0"],
                          columns="Method", values="Coverage")
             .reset_index()
             .sort_values("R0"))
table.to_csv(out, index=False)
print(f"\nwrote {out}  ({len(table)} rows)")
print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
