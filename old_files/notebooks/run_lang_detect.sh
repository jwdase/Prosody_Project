#!/bin/bash
#SBATCH --job-name=lang_detect
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00

# -- load your personal conda install --
# adjust this path if your Conda lives elsewhere
source /home/jwdase/miniconda3/etc/profile.d/conda.sh

# now activate your env
conda activate landD

echo "Running on $(hostname)"
echo "Using Python from: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

python Multi_Language_Detection.py

