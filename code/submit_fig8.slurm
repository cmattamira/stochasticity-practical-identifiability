#!/bin/bash
#SBATCH -A YOUR_ACCOUNT            # <-- your allocation
#SBATCH -J fig8
#SBATCH --partition=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-18:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#
# Figure 8: coverage of the Gaussian-noise method across a grid of noise
# levels sigma, one array task per line of fig8_manifest.tsv (line 1 is a
# comment, so task N reads line N+1). Each task runs case1.py at a single
# sigma for a single parameter combination.
#
# The manifest has three blocks:
#     tasks  1-12   sigma = 0.1 and 0.2 for the six combinations
#     tasks 13-40   the sigma values needed to locate where coverage reaches
#                   the nominal level, plus the whole grid for the right panel
#     tasks 41-60   the rest of the grid (not needed for the figure)
#
# At M = 1000 the sigma = 0.1 and 0.2 values already exist in results/case1,
# so only tasks 13-40 are needed:
#     sbatch --array=13-40%28 submit_fig8.slurm
# At M = 100 nothing exists yet, so run tasks 1-40:
#     sbatch --array=1-40%40 --export=ALL,BOOTSTRAP=100 --time=0-03:00:00 submit_fig8.slurm
#
# Timing on one core: about 0.1 s per fit, so roughly 20 minutes per task at
# M = 100 and 3 to 5 hours per task at M = 1000.

set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate identifiability          # <-- your env name

# Thousands of tiny ODE solves; threaded BLAS only causes contention here.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

BASE=$HOME/identifiability              # <-- where you cloned the repository
CODE=$BASE/code
DATA=$BASE/data

BOOTSTRAP=${BOOTSTRAP:-1000}
RESULTS=${RESULTS:-$BASE/results/fig8_M$BOOTSTRAP}

# Line 1 of the manifest is a header comment, so skip it.
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
read -r COMBO SIGMA SEED ALPHA BETA R0 < <(sed -n "${LINE}p" "$CODE/fig8_manifest.tsv")

# Fold the bootstrap count into the seed so the M = 100 and M = 1000 runs of
# the same task do not reuse the same random stream.
SEED=$(( SEED + BOOTSTRAP ))

mkdir -p "$RESULTS" logs

echo "task=$SLURM_ARRAY_TASK_ID combo=$COMBO sigma=$SIGMA alpha=$ALPHA beta=$BETA R0=$R0"
echo "bootstrap=$BOOTSTRAP seed=$SEED host=$(hostname) start=$(date -Is)"

python "$CODE/case1.py" \
    --combo "$COMBO" \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --sigmas "$SIGMA" \
    --data "$DATA" \
    --out "$RESULTS"

echo "end=$(date -Is)"
