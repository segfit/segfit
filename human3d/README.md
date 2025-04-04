## Human3D


To run the model, use the following script after passing the path to the checkpoint as an argument.

```
python infer_mhbps.py general.checkpoint='pretrained/FSK.ckpt'
```

There are a couple of demo scenes, `frame_01661.ply` and `frame_standing_depth_scan.ply` in the folder. Right now we are not passing the path to the scene as an argument but instead reading them within the `infer_mhbps.py` script. Comment in/out the scene to experiment with the model. On another note, the model expects point clouds in z-up coordinate system, there is also a function to convert a y-up scene to a z-up scene. If you'd like to use it, don't forget to set the `is_z_up` parameter.