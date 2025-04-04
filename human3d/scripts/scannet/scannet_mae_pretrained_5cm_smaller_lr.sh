#!/bin/bash
#SBATCH --job-name=5cm_pret
#SBATCH --output=../../logs/scannet_5cm_mae_pretrained_%j.out
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

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="5cm_mae_pretrained_debug_lr_001" \
general.eval_on_segments=true \
general.train_on_segments=true \
general.backbone_checkpoint="/globalwork/nekrasov/shared/pretrained_scannet_5cm.pth" \
data.voxel_size=0.05 \
trainer.check_val_every_n_epoch=1 \
optimizer.lr=0.00001
