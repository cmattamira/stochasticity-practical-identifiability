#!/bin/bash
#SBATCH -A YOUR_ACCOUNT            # <-- your allocation
#SBATCH -J coverage
#SBATCH --partition=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-18:00:00
#SBATCH --array=1-48%12
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#
# Coverage across the 16 parameter combinations, one array task per line of
# manifest.tsv (line 1 is a comment, so task N reads line N+1).
#
# Submit at the default M = 1000, which is what the manuscript states:
#     sbatch submit_coverage.slurm
# or, to reproduce the old numbers at M = 100:
#     sbatch --export=ALL,BOOTSTRAP=100 submit_coverage.slurm
#
# Case 3c, the CTMC arm with major-outbreak conditioning, is tasks 49-64:
#     sbatch --array=49-64%12 submit_coverage.slurm
#
# To keep the bootstrap clouds as well as the coverage numbers, point CLOUDS
# at a directory and send the results somewhere separate so the production
# numbers are not overwritten:
#     sbatch --export=ALL,RESULTS=$HOME/identifiability/results_clouds,\
# CLOUDS=$HOME/identifiability/clouds --array=3,5,15,16 submit_coverage.slurm
# The seeds come from the manifest either way, so a cloud run reproduces the
# coverage number the production run already gave for that task. That is worth
# checking when the first one lands: it confirms the clouds belong to the
# numbers in the table.
#
# Timing, measured on one core (case1 combo 2, scaled linearly in M):
#     M = 100   longest task ~1 h    -> ~4 h wall clock at 12-way
#     M = 1000  longest task ~9.5 h  -> ~37 h wall clock at 12-way
# The 18 h walltime above covers M = 1000 with margin for a slower core.

set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate identifiability          # <-- your env name

# Thousands of tiny ODE solves; threaded BLAS only causes contention here.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

BASE=$HOME/identifiability              # <-- where you cloned the repository
CODE=$BASE/code
DATA=$BASE/data

# Both can be overridden with --export, which is how a cloud-saving run keeps
# clear of the production results.
RESULTS=${RESULTS:-$BASE/results}
CLOUDS=${CLOUDS:-}

BOOTSTRAP=${BOOTSTRAP:-1000}

# Line 1 of the manifest is a header comment, so skip it.
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
read -r CASE COMBO SEED ALPHA BETA R0 < <(sed -n "${LINE}p" "$CODE/manifest.tsv")

# Fold the bootstrap count into the seed so an M=100 run and an M=1000 run of
# the same task do not reuse the same random stream.
SEED=$(( SEED + BOOTSTRAP ))

OUT=$RESULTS/$CASE
mkdir -p "$OUT" logs

# Case 4 is the only arm that takes extra arguments: it needs to know where its
# scaling calibration lives, and whether to couple the scaling factor to the
# shift. The other three take none, so EXTRA stays empty for them.
#
#   SCALING_ROUTE   parametric (default) reads summary_N1000.csv and
#                   interpolates at each fitted pair's own R0; empirical
#                   resamples the measured values for the cell.
#   RHO             correlation between scaling factor and shift, default 0.
#                   Parametric route only.
EXTRA=""
if [ "$CASE" = "case4" ]; then
    ROUTE=${SCALING_ROUTE:-parametric}
    if [ "$ROUTE" = "parametric" ]; then
        EXTRA="--scaling-route parametric --scaling-summary $DATA/fig7/summary_N1000.csv --rho ${RHO:-0}"
    else
        EXTRA="--scaling-route empirical --scaling-samples $DATA/fig7/samples"
    fi
fi

echo "task=$SLURM_ARRAY_TASK_ID case=$CASE combo=$COMBO alpha=$ALPHA beta=$BETA R0=$R0"
echo "bootstrap=$BOOTSTRAP seed=$SEED host=$(hostname) start=$(date -Is)"
echo "results=$OUT clouds=${CLOUDS:-none} extra=${EXTRA:-none}"

# Two spellings of the same command rather than one clever one, so it is
# obvious what runs in each case. EXTRA is deliberately unquoted: it is either
# empty or several separate arguments.
if [ -n "$CLOUDS" ]; then
    mkdir -p "$CLOUDS/$CASE"
    python "$CODE/$CASE.py" \
        --combo "$COMBO" \
        --bootstrap "$BOOTSTRAP" \
        --seed "$SEED" \
        --data "$DATA" \
        --out "$OUT" \
        --save-clouds "$CLOUDS/$CASE" $EXTRA
else
    python "$CODE/$CASE.py" \
        --combo "$COMBO" \
        --bootstrap "$BOOTSTRAP" \
        --seed "$SEED" \
        --data "$DATA" \
        --out "$OUT" $EXTRA
fi

echo "end=$(date -Is)"
