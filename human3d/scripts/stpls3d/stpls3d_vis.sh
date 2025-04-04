#!/bin/bash
#SBATCH --job-name=vSTPLS3D
#SBATCH --output=../../logs/vis_stpls3d_%A-%a.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=tui

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_TOPK=750
CURR_QUERY=160
CURR_SIZE=54

cd ../../
python main_instance_segmentation.py \
general.experiment_name="debug3_vis_stpls3d_cropsize_${CURR_SIZE}_query_${CURR_QUERY}_topk_${CURR_TOPK}" \
general.project_name="debug" \
data/datasets=stpls3d \
data/data_loaders=simple_loader_save_memory \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="/globalwork/schult/checkpoints_icra/stpls3d/stpls3d_val_clipped.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.filter_out_instances=true \
general.scores_threshold=0.4 \
general.save_visualizations=true \
general.visualization_point_size=300
