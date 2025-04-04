#!/usr/local_rwth/bin/zsh
#SBATCH --partition=c18g
#SBATCH --account=rwth1261
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80GB
#SBATCH --job-name=HB06_Res18_36epochs
#SBATCH --time=5-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --mail-user=schult@vision.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --output=../../../logs/HB06_Res18_36epochs_%j.txt

export OMP_NUM_THREADS=3

source /home/js411977/.zshrc

module load cuda/11.6

conda deactivate
conda activate mask3d
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

cd ../../../
python main_instance_segmentation.py \
  general.experiment_name="HB06_Res18_36epochs_fix2" \
general.project_name="human_instance" \
data/datasets=behave \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d \
loss=set_criterion \
model.num_queries=5 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=12 \
general.reps_per_epoch=1 \
general.save_visualizations=false \
model.config.backbone._target_=models.Res16UNet18B \
data.part2human=true \
loss.num_classes=2 \
model.num_classes=2 \
callbacks=callbacks_instance_segmentation_human
