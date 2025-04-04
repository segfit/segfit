#!/bin/bash
#SBATCH --job-name=HE06
#SBATCH --output=../../../logs/HE06_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=a40-hi

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../../
python main_instance_segmentation.py \
general.experiment_name="HE06" \
general.project_name="human_instance" \
data/datasets=egobody \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d \
loss=set_criterion \
model.num_queries=5 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=12 \
general.reps_per_epoch=1 \
general.save_visualizations=false \
model.config.backbone._target_=models.Res16UNet18B \
data.part2human=true \
loss.num_classes=2 \
model.num_classes=2 \
callbacks=callbacks_instance_segmentation_human

