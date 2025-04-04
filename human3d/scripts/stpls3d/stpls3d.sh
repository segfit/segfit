#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_TOPK=750
CURR_QUERY=160
CURR_SIZE=54

python main_instance_segmentation.py \
general.experiment_name="stpls3d" \
general.project_name="stpls3d" \
data/datasets=stpls3d \
data/data_loaders=simple_loader_save_memory \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.crop_length=50.0 \
data.crop_min_size=0 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
data.is_scannet=false \
data.is_stpls3d=true \
general.reps_per_epoch=100 \
model.num_queries=100 \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B

python main_instance_segmentation.py \
general.experiment_name="stpls3d_cropsize_${CURR_SIZE}_query_${CURR_QUERY}_topk_${CURR_TOPK}" \
general.project_name="stpls3d_eval" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
data.is_scannet=false \
data.is_stpls3d=true \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="/globalwork/schult/checkpoints_icra/stpls3d/stpls3d_val_clipped.ckpt" \
data.crop_length=${CURR_SIZE} \
general.stpls3d_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK}
