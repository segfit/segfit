conda activate human3d-demo #activate environment

#demo with point-cloud generated from the realsense
python api_human3d_no_depth.py #launch server
python vis_gui_realsense_pcd.py #launch demo gui

#demo with rgb captured from the sensor
python api_human3d.py #launch the server
python vis_gui_realsense_rgb.py #launch demo gui

#demo with rgb caputured from webcam
python api_human3d.py #launch the server
python vis_gui.py #launch demo gui

