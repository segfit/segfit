#!/bin/bash
#SBATCH --job-name=02_humseg
#SBATCH --output=../../logs/02_humseg_baseline_24epochs_%j.out
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
  general.project_name="human_segmentation" \
  general.experiment_name="02_baseline_humseg_24epochs_nt2" \
  data/datasets=human_segmentation \
  general.num_targets=2 \
  data.num_labels=2 \
  model.num_queries=20 \
  trainer.check_val_every_n_epoch=1 \
  general.topk_per_image=-1 \
  model.non_parametric_queries=false \
  trainer.num_sanity_val_steps=10 \
  +trainer.limit_val_batches=400 \
  trainer.max_epochs=24
