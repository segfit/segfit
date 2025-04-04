#!/bin/bash
#SBATCH --job-name=SK06_no_restricted_ca_no_two_stage
#SBATCH --output=../../../logs/SK06_no_restricted_ca_no_two_stage_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=kloud

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../../
python main_instance_segmentation.py \
general.experiment_name="SK06_no_restricted_ca_no_two_stage_2" \
general.project_name="multi_human_parsing" \
data/datasets=synthetic_kinect \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d_hp \
loss=set_criterion_hp_no_two_stage \
model.num_human_queries=5 \
model.num_parts_per_human_queries=16 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
general.save_visualizations=false \
model.config.backbone._target_=models.Res16UNet18B \
model.restricted_cross_attention=false
