#!/bin/bash
#SBATCH --job-name=prepHuman
#SBATCH --output=prep_humanseg_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --gres=gpu:0
#SBATCH --time=5-00:00:00
#SBATCH --partition=a40-lo

python humanseg_preprocessing.py preprocess

