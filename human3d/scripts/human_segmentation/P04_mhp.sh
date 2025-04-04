#!/bin/bash
#SBATCH --job-name=P04_mhp
#SBATCH --output=../../logs/P04_mhp_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:3090:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=tui

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../
python main_instance_segmentation.py \
  general.project_name="human_segmentation" \
  general.experiment_name="P04_mhp_fix" \
  data/datasets=human_part_segmentation \
  general.num_targets=27 \
  data.num_labels=27 \
  model=mask3d_hp \
  loss=set_criterion_hp \
  model.num_queries=280 \
  trainer.check_val_every_n_epoch=25 \
  general.topk_per_image=-1 \
  model.non_parametric_queries=false \
  trainer.num_sanity_val_steps=0 \
  trainer.max_epochs=24 \
  data.batch_size=10 \
  data.num_workers=10 \
  +trainer.limit_val_batches=400
