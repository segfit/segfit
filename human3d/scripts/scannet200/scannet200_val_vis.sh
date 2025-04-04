#!/bin/bash
#SBATCH --job-name=bScanNet200
#SBATCH --output=../../logs/vis_scannet200_benchmark_%A-%a.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --partition=bomonti

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=750
CURR_QUERY=150

cd ../../
python main_instance_segmentation.py \
general.experiment_name="debug2_vis_scannet200_benchmark_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
general.project_name="debug" \
general.checkpoint="/globalwork/schult/checkpoints_icra/scannet200/scannet200_benchmark_clipped.ckpt" \
data/datasets=scannet200 \
general.num_targets=201 \
data.num_labels=200 \
general.eval_on_segments=true \
general.train_on_segments=true \
general.train_mode=false \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
general.filter_out_instances=true \
general.scores_threshold=0.4 \
general.save_visualizations=true \
data.test_mode=test
