cd ../../

python main_instance_segmentation.py \
general.experiment_name="H01_vis" \
general.project_name="debug" \
data/datasets=egobody \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d_hp \
loss=set_criterion_hp \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.num_sanity_val_steps=1000 \
+trainer.limit_val_batches=1000 \
trainer.max_epochs=40 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
general.save_visualizations=true \
general.train_mode=false \
general.checkpoint="/globalwork/schult/H01.ckpt" \
model.num_human_queries=5 \
model.num_parts_per_human_queries=16

