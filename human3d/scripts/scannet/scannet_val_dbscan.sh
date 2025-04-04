#!/bin/bash
#SBATCH --job-name=scannet05
#SBATCH --output=../../logs/scannet_dbscan05_%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --partition=a40-lo

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.6
CURR_TOPK=500
CURR_QUERY=150

# TRAIN
cd ../../
python main_instance_segmentation.py \
general.experiment_name="debugkhkkkhjjjhhj" \
general.project_name="debug" \
general.checkpoint="/globalwork/schult/checkpoints_icra/scannet/scannet_val_clipped.ckpt" \
general.train_mode=false \
general.eval_on_segments=true \
general.train_on_segments=true \
model.num_queries=150 \
general.topk_per_image=500 \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}

