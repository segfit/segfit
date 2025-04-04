#!/bin/bash
#SBATCH --job-name=clipScanNet
#SBATCH --output=../../logs/clip_init_baseline_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=6
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=lopri

export OMP_NUM_THREADS=2  # speeds up MinkowskiEngine

cd ../../
python main_instance_segmentation.py \
general.experiment_name="clip_init_baseline" \
general.project_name="scannet" \
general.eval_on_segments=true \
general.train_on_segments=true \
data.voxel_size=0.05 \
general.add_clip=true \
model.query_init="clip_init"
