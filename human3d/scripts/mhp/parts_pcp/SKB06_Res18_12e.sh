#!/bin/bash
#SBATCH --job-name=SKB06_Res18_12e
#SBATCH --output=../../../logs/SKB06_Res18_12e_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=kloud

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../../
python main_instance_segmentation.py \
general.experiment_name="SKB06_Res18_12e" \
general.project_name="multi_human_parsing" \
data/datasets=synthetic_kinect_behave \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d_hp \
loss=set_criterion_hp \
model.num_human_queries=5 \
model.num_parts_per_human_queries=16 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=12 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
general.save_visualizations=false \
model.config.backbone._target_=models.Res16UNet18B
