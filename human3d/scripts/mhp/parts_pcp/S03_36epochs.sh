#!/bin/bash
#SBATCH --job-name=S03_36epochs
#SBATCH --output=../../../logs/S03_36epochs_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=6
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=lopri

export OMP_NUM_THREADS=2  # speeds up MinkowskiEngine

cd ../../../
python main_instance_segmentation.py \
general.experiment_name="S03_36epochs_fix" \
general.project_name="multi_human_parsing" \
data/datasets=human_part_segmentation \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d_hp \
loss=set_criterion_hp \
model.num_human_queries=5 \
model.num_parts_per_human_queries=16 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=3 \
general.reps_per_epoch=1 \
general.save_visualizations=false
