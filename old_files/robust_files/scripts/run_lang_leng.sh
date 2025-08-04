#!/bin/bash
#SBATCH --job-name=lang_detect
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00

# Load personal conda install
source /om2/user/jwdase/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate torchenv

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to parent directory (where the Python file is)
cd ..

echo "Running on $(hostname)"
echo "Using Python from: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run the Python script from the parent directory
python language_lengths.py
