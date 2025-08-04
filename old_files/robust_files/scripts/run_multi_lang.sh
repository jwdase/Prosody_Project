#!/bin/bash
#SBATCH --job-name=lang_detect
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --time=06:00:00

# Load personal conda install
source /om2/user/jwdase/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate torchgpu

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to parent directory (where the Python file is)
cd ..

echo "Running on $(hostname)"
echo "Using Python from: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run the Python script from the parent directory
python -u testing_fourier_transform.py