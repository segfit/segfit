#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

python main_instance_segmentation.py \
  general.project_name="human_segmentation_debug" \
  general.experiment_name="debug" \
  data/datasets=human_segmentation \
  general.num_targets=2 \
  data.num_labels=1 \
  general.reps_per_epoch=100 \
  trainer.check_val_every_n_epoch=10
