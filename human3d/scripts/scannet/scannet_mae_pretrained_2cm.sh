#!/bin/bash
#SBATCH --job-name=2cm_pret
#SBATCH --output=../../logs/scannet_2cm_mae_pretrained_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=a40-hi

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="2cm_mae_pretrained_001" \
general.eval_on_segments=true \
general.train_on_segments=true \
general.backbone_checkpoint="/globalwork/nekrasov/shared/pretrained_scannet_2cm.pth" \
trainer.check_val_every_n_epoch=1
