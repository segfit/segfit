import pdb

# Library imports
import os
import uuid
import shutil
import threading
from http import HTTPStatus
from typing import Dict

from flask import Flask
from flask.globals import request
from flask.wrappers import Response
import gzip

import torch
import orjson
import sys

from api_human3d_pipeline import create_human3d_pipeline
from api_depth_pipeline import create_depth_pipeline
from api_types import *
import hydra
from omegaconf import DictConfig
import time
upload_dir = "./upload"

pipeline_depth = None
pipeline_human3d = None

init_params = {
    "img_dir": upload_dir,
    "res_dir": os.path.join(upload_dir, "preds"),
    "is_input_z_up": True,
    "filter_depth": False, #True
    "filter_pcd_outliers": True,
    "filter_depth_far_plane": 10.0,
    "write_pcd": False,
    "save_viz": False,
    "image_name": "image_rgb.jpg",
    'min_conf_score':0.89, 
    'min_num_points':2000
}

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation_demo.yaml")
def init_model(cfg: DictConfig):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if not os.path.exists(init_params["res_dir"]):
        os.makedirs(init_params["res_dir"])
    
    global pipeline_human3d

    pipeline_human3d = create_human3d_pipeline(cfg) #, 0.89, 2000)

    global pipeline_depth
    pipeline_depth = create_depth_pipeline(filter_depth=init_params["filter_depth"], filter_pcd_outliers=init_params["filter_pcd_outliers"], filter_depth_far_plane=init_params["filter_depth_far_plane"])

    print("Initialization complete, model is available")
    
# Init model
init_thread = threading.Thread(target=init_model, name="init")
init_thread.start()
#init_model()

# Init app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()
    curr_uuid = str(uuid.uuid4())
    upload_dir_uuid = os.path.join(upload_dir, curr_uuid)

    while os.path.exists(upload_dir_uuid):
        curr_uuid = uuid.uuid4()
        upload_dir_uuid = os.path.join(upload_dir, curr_uuid)

    upload_dir_uuid_src = os.path.join(upload_dir_uuid, "src")
    upload_dir_uuid_res = os.path.join(upload_dir_uuid, "res")
    if not os.path.exists(upload_dir_uuid_src):
        os.makedirs(upload_dir_uuid_src)
    if not os.path.exists(upload_dir_uuid_res):
        os.makedirs(upload_dir_uuid_res)

    DataProcess.from_multipart(request, param_dict={"input_dir": upload_dir_uuid_src, "image_name": init_params["image_name"]})
    
    result = None
    pcd_coords = pipeline_depth.run_pipeline(
        src_dir=upload_dir_uuid_src,
        res_dir=upload_dir_uuid_res,
        write_pcd=init_params['write_pcd'],
        image_name=init_params['image_name']
    )

    pred_inst, pred_parts, pred_scores, full_coords = pipeline_human3d.run_pipeline(pcd_coords,
        src_dir=upload_dir_uuid_src,
        res_dir=upload_dir_uuid_res,
        is_input_z_up=init_params['is_input_z_up'],
        save_viz=init_params['save_viz'],
        min_conf_score=init_params['min_conf_score'],
        min_num_points=init_params['min_num_points']
    )

    result = {"pred_inst":pred_inst.tolist(),"pred_parts": pred_parts.tolist(),"pred_scores": pred_scores.tolist(),"full_coords": np.round(full_coords, decimals=2).tolist()}
        
    torch.cuda.empty_cache()
    print('Cuda cache emptied!')

    result = gzip.compress(orjson.dumps(result), compresslevel=5)

    shutil.rmtree(upload_dir_uuid) #if this is commented out, the folder will not be deleted
    #print('UPLOAD DIR: ', upload_dir_uuid)

    end = time.perf_counter()
    print(f"Overall time: {end - start:0.4f} seconds.")
    print(Response(response=result, status=HTTPStatus.OK, mimetype="application/json"))
    return Response(response=result, status=HTTPStatus.OK, mimetype="application/json")


port_id = 5000
print("API server is listening on {}".format(port_id))

import signal



if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=port_id)
    
    reload = True

    def handler(signum, frame):
        global reload
        reload = False
        # clean-up uploaded files
        del_folders = [os.path.join(upload_dir, el) for el in os.listdir(upload_dir) if el not in ['img_write', 'preds'] and os.path.isdir(os.path.join(upload_dir, el))]
        for folder in del_folders:
            shutil.rmtree(folder)
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    while reload:
        try:
            app.run(host='0.0.0.0', port=port_id)          
        except:
            if reload:
                print('[INFO]: Server crashed, restarting...')
            else:
                print('[INFO]: Stopping the server...')
                sys.exit(0)

