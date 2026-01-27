#!/bin/bash -l
#SBATCH --job-name=lang_detect
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=ou_bcs_low
#SBATCH --mem=128G
#SBATCH --time=20:00:00

module load matlab

echo "Loaded Matlab"
echo which matlab

matlab -nodisplay -r "run_many_high_performance('data/it.txt')"