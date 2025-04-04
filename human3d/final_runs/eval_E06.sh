cd ..
python main_instance_segmentation.py \
general.experiment_name="E06_eval" \
general.project_name="final_human3d" \
data/datasets=egobody \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d_hp \
loss=set_criterion_hp \
model.num_human_queries=5 \
model.num_parts_per_human_queries=16 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
model.config.backbone._target_=models.Res16UNet18B \
general.checkpoint="/work2/human3d_final_ckpts/humanseg_final_ckpts/mhp/iccv/E06.ckpt" \
general.train_mode=false \
general.save_visualizations=false
