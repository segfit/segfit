#!/bin/bash
#SBATCH --job-name=PE06
#SBATCH --output=../../../logs/PE06_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:3090:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=ozujsko

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../../
python main_instance_segmentation.py \
general.experiment_name="PE06_2" \
general.project_name="body_part_instance_segmentation" \
data/datasets=egobody \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d \
loss=set_criterion \
model.num_queries=80 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
general.save_visualizations=false \
model.config.backbone._target_=models.Res16UNet18B \
general.body_part_segmentation=true \
callbacks=callbacks_instance_segmentation_body_part_instance_segmentation
