#!/bin/bash -l
#SBATCH --job-name=lang_detect
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=ou_bcs_low
#SBATCH --mem=128G
#SBATCH --time=20:00:00

cd ..
cd ..

source .venv/bin/activate

cd src

PYTHONPATH=. python -i scripts/lengths.py