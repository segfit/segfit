import os
import cv2
import numpy as np
import requests
import json
import pdb
from api_viz_dicts import map2color_instances, map2color_parts
import open3d as o3d
from imutils.video import VideoStream
import time
import gzip
import orjson

url = 'http://localhost:5000/predict'  # Replace with your server's endpoint

img_write_dir = 'upload/img_write'
if not os.path.exists(img_write_dir):
    os.makedirs(img_write_dir)
image_write_path = os.path.join(img_write_dir, 'image_rgb.jpg')

# TO-DO: put your own K matrix here
K_orig_path = os.path.join(img_write_dir, 'K_orig.txt')
if not os.path.exists(K_orig_path):
    raise ValueError("Please put your own intrinsics matrix in the img_write directory")


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Human3D', visible=True)
ctr = vis.get_view_control()
render_opt = vis.get_render_option()
#render_opt.point_show_normal = True
print(render_opt.point_size)
render_opt.point_size = 10

parameters = o3d.io.read_pinhole_camera_parameters("ScreenCameraDemo.json")
#pdb.set_trace()
#K = np.loadtxt(K_orig_path)
#parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, K[0,0], K[1,1], K[0,2], K[1,2])

print('---')

#pdb.set_trace()
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
frame_bgr = vs.read()
key = cv2.waitKey(1000) & 0xFF

counter = 0
while True:
    print("CAPTURED")
    tic = time.perf_counter()
    vis.clear_geometries()

    frame_bgr = vs.read()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_write_path, cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    files = {'image_rgb':open(image_write_path, 'rb'), 'K_orig':open(K_orig_path, 'rb')}
    response = requests.post(url, files=files)

    response_dict = orjson.loads(gzip.decompress(response.content)) #dict_keys(['pred_inst', 'pred_parts', 'full_coords'])

    pred_inst = np.asarray(response_dict['pred_inst'])
    pred_parts = np.asarray(response_dict['pred_parts'])
    full_coords = np.asarray(response_dict['full_coords'])

    tac = time.perf_counter()
    print(f"Time elapsed: {tac - tic:0.4f} seconds")
    
    T_z_up_to_y_up = np.array([ [-1., 0., 0., 0.], 
                                [0., 0., 1., 0.], 
                                [0., -1., 0., 0.], 
                                [0., 0., 0., 1.]]) 
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_coords)


    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
    #pcd.estimate_normals()
    inst_colors = (np.asarray(map2color_instances(pred_inst)).T)/255.
    pcd.colors = o3d.utility.Vector3dVector(inst_colors)
    #o3d.io.write_point_cloud(os.path.join(img_write_dir, 'pred_inst.ply'), pcd)
    #part_colors = (np.asarray(map2color_parts(pred_parts)).T)/255.
    #pcd.colors = o3d.utility.Vector3dVector(part_colors)
    #o3d.io.write_point_cloud(os.path.join(img_write_dir, 'pred_part.ply'), pcd)
    


    # TO-DO: visualizer coordinate system should be different
    #pdb.set_trace()
    pcd = pcd.transform(T_z_up_to_y_up)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()


    #print(vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic)
    #ctr.convert_from_pinhole_camera_parameters(vis_cam, allow_arbitrary=True)
    #print(vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix)


    vis.poll_events()
    vis.update_renderer()
    #vis.run()

    #pdb.set_trace()
    #key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(5) & 0xFF
    #counter +=1
    #if counter == 2:
    #    break
    
    if key == ord("q"):
        break

    tac_final = time.perf_counter()
    print(f"Iter time elapsed: {tac_final - tic:0.4f} seconds")

key = cv2.waitKey(30) & 0xFF
cv2.destroyAllWindows()
vs.stop()
#vis.destroy_window()
