#!/usr/local_rwth/bin/zsh
#SBATCH --partition=c18g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180GB
#SBATCH --job-name=s3dis_wo_dbscan_ablation
#SBATCH --time=0-06:00:00
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --output=../../logs/s3dis_pretrained_wo_dbscan_ablation_%A-%a.txt
#SBATCH --array=1

export OMP_NUM_THREADS=3
source /home/js411977/.zshrc

module load cuda/11.6

conda deactivate
conda activate mask3d

CURR_AREA=5  # set the area number accordingly [1,6]
CURR_TOPK=-1
CURR_QUERY=100

cd ../../
python main_instance_segmentation.py \
general.project_name="s3dis_eval" \
general.experiment_name="area${CURR_AREA}_pretrained_topk_${CURR_TOPK}_q_${CURR_QUERY}" \
general.checkpoint="/work/js411977/s3dis_area5_scannet_pretrained_clipped.ckpt" \
general.train_mode=false \
data.batch_size=4 \
data/datasets=s3dis \
general.num_targets=14 \
data.num_labels=13 \
general.area=${CURR_AREA} \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK}
