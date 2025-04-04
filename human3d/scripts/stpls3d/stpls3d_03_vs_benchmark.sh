#!/usr/local_rwth/bin/zsh
#SBATCH --partition=c18g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --job-name=b_stpls3d_03_vs
#SBATCH --time=5-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --output=../../logs/benchmark_stpls3d_03_vs_%j.txt

export OMP_NUM_THREADS=3

source /home/js411977/.zshrc

module load cuda/11.6

conda deactivate
conda activate mask3d
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../
python main_instance_segmentation.py \
general.experiment_name="stpls3d_val_03_vs_benchmark" \
general.project_name="stpls3d" \
data/datasets=stpls3d \
data/data_loaders=simple_loader_save_memory \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.3 \
data.crop_length=50.0 \
data.crop_min_size=0 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=100 \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
data.train_mode=train_validation
