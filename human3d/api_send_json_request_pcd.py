
import os
import cv2
import numpy as np
import requests
import orjson
import gzip
import pdb
from api_viz_dicts import map2color_instances, map2color_parts
import open3d as o3d

url = 'http://localhost:5001/predict'  # Replace with your server's endpoint

# Load the image and the intrinsics
#image_path = 'upload/img_write/image_rgb.jpg'
#K_orig_path = 'upload/img_write/K_orig.txt'

scene_pcd = 'upload/egobody_example/scene_pcd_orig.ply'
ply_write_dir = 'upload/egobody_example'

files = {'scene_pcd':open(scene_pcd, 'rb')}
response = requests.post(url, files=files)
response_dict = orjson.loads(gzip.decompress(response.content)) #dict_keys(['pred_inst', 'pred_parts', 'full_coords'])
pred_inst = np.asarray(response_dict['pred_inst'])
pred_parts = np.asarray(response_dict['pred_parts'])
full_coords = np.asarray(response_dict['full_coords'])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(full_coords)
#pcd.estimate_normals()
inst_colors = (np.asarray(map2color_instances(pred_inst)).T)/255.
pcd.colors = o3d.utility.Vector3dVector(inst_colors)
o3d.io.write_point_cloud(os.path.join(ply_write_dir, 'pred_inst_from_pcd.ply'), pcd)
part_colors = (np.asarray(map2color_parts(pred_parts)).T)/255.
pcd.colors = o3d.utility.Vector3dVector(part_colors)
o3d.io.write_point_cloud(os.path.join(ply_write_dir, 'pred_part_from_pcd.ply'), pcd)
