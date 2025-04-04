#!/usr/local_rwth/bin/zsh
#SBATCH --partition=c18g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=90GB
#SBATCH --job-name=14_eval
#SBATCH --time=5-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --output=../../logs/14_eval_stpls3d_%A-%a.txt
#SBATCH --array=1-4

export OMP_NUM_THREADS=3

source /home/js411977/.zshrc

module load cuda/11.6

conda deactivate
conda activate mask3d

PARAM_QUERIES=(160 200 250 300)

CURR_QUERY=${PARAM_QUERIES[SLURM_ARRAY_TASK_ID]}
CURR_TOPK=750
CURR_SIZE=54

cd ../../
python main_instance_segmentation.py \
general.experiment_name="14_stpls3d_cropsize_${CURR_SIZE}_query_${CURR_QUERY}_topk_${CURR_TOPK}_fix" \
general.project_name="stpls3d_eval" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
model.positional_encoding_type="sine" \
model.use_level_embed=true \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="/work/js411977/14_stpls3d.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK}
