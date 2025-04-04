#!/bin/bash
#SBATCH --job-name=pS3DIS
#SBATCH --output=../../logs/vis_s3dis_pretrained_%A-%a.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=a40-lo
#SBATCH --array=1-6

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_AREA=${SLURM_ARRAY_TASK_ID}  # set the area number accordingly [1,6]
CURR_DBSCAN=0.6
CURR_TOPK=-1
CURR_QUERY=100

cd ../../
python main_instance_segmentation.py \
general.project_name="debug" \
general.experiment_name="debug3_vis_area${CURR_AREA}_scannet_pretrained_eps_${CURR_DBSCAN}_topk_${CURR_TOPK}_q_${CURR_QUERY}" \
general.checkpoint="/globalwork/schult/checkpoints_icra/s3dis/scannet_pretrained/area${CURR_AREA}_clipped.ckpt" \
general.train_mode=false \
data.batch_size=4 \
data/datasets=s3dis \
general.num_targets=14 \
data.num_labels=13 \
general.area=${CURR_AREA} \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
general.filter_out_instances=true \
general.scores_threshold=0.4 \
general.save_visualizations=true
