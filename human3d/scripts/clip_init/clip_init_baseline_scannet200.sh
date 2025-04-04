#!/bin/bash
#SBATCH --job-name=clipScanNet200
#SBATCH --output=../../logs/clip_init_baseline_scannet200_%j.out
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
general.experiment_name="clip_init_baseline_scannet200_fix2" \
general.project_name="scannet200" \
general.eval_on_segments=true \
general.train_on_segments=true \
data/datasets=scannet200 \
general.num_targets=201 \
data.num_labels=200 \
data.voxel_size=0.05 \
general.add_clip=true \
model.query_init="clip_init"
