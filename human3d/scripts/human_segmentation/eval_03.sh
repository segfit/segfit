#!/bin/bash
#SBATCH --job-name=e03_humseg
#SBATCH --output=../../logs/eval_03_%j.out
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
  general.project_name="human_segmentation_eval" \
  general.experiment_name="eval_03_001" \
  data/datasets=human_segmentation \
  general.num_targets=2 \
  data.num_labels=2 \
  model.num_queries=20 \
  trainer.check_val_every_n_epoch=25 \
  general.topk_per_image=-1 \
  model.non_parametric_queries=false \
  trainer.num_sanity_val_steps=0 \
  trainer.max_epochs=24 \
  data.batch_size=10 \
  data.num_workers=10 \
  general.checkpoint="/home/schult/projects/release/Mask3D/saved/03_24epochs_10bs_nt2/last.ckpt" \
  general.train_mode=false
