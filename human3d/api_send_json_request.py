
import os
import cv2
import numpy as np
import requests
import orjson
import gzip
import pdb

url = 'http://localhost:5000/predict'  # Replace with your server's endpoint

# Load the image and the intrinsics
image_path = 'upload/preds/debug_scene/src/image_rgb.jpg'
K_orig_path = 'upload/preds/debug_scene/src/K_orig.txt'

files = {'image_rgb':open(image_path, 'rb'), 'K_orig':open(K_orig_path, 'rb')}
response = requests.post(url, files=files)
response_dict = orjson.loads(gzip.decompress(response.content)) #dict_keys(['pred_inst', 'pred_parts', 'full_coords'])
pred_inst = np.asarray(response_dict['pred_inst'])
pred_parts = np.asarray(response_dict['pred_parts'])
full_coords = np.asarray(response_dict['full_coords'])

