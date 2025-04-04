#!/bin/bash
#SBATCH --job-name=wo_s3dis
#SBATCH --output=../../logs/s3dis_wo_dbscan_ablation_%A-%a.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --partition=a40-lo
#SBATCH --array=0

export OMP_NUM_THREADS=3

CURR_AREA=5  # set the area number accordingly [1,6]
CURR_TOPK=-1
CURR_QUERY=100

cd ../../
python main_instance_segmentation.py \
general.project_name="s3dis_eval" \
general.experiment_name="area${CURR_AREA}_pretrained_topk_${CURR_TOPK}_q_${CURR_QUERY}_fix2" \
general.checkpoint="/globalwork/schult/checkpoints_icra/s3dis/scannet_pretrained/area5_clipped.ckpt" \
general.train_mode=false \
data.batch_size=4 \
data/datasets=s3dis \
general.num_targets=14 \
data.num_labels=13 \
general.area=${CURR_AREA} \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK}
