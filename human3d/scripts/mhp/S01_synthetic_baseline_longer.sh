#!/bin/bash
#SBATCH --job-name=S01_synthetic_baseline_longer_85q
#SBATCH --output=../../logs/S01_synthetic_baseline_longer_85q_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:3090:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=brewdog

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../

# TRAIN
python main_instance_segmentation.py \
general.project_name="mhp_synthetic" \
general.experiment_name="S01_synthetic_baseline_longer_85q_fix" \
data/datasets=human_part_segmentation \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d_hp \
loss=set_criterion_hp \
model.num_queries=85 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.num_sanity_val_steps=400 \
+trainer.limit_val_batches=400 \
trainer.max_epochs=40 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
general.save_visualizations=false
