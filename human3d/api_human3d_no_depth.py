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


from api_human3d_pipeline import create_human3d_pipeline
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
    "filter_depth": True,
    "filter_pcd_outliers": True,
    "write_pcd": False,
    "save_viz": False,
    "image_name": "image_rgb.jpg",
    "pcd_name": "scene_pcd.ply",
    'min_conf_score':0.86, 
    'min_num_points':8000
}

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation_demo.yaml")
def init_model(cfg: DictConfig):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if not os.path.exists(init_params["res_dir"]):
        os.makedirs(init_params["res_dir"])
    
    global pipeline_human3d
    pipeline_human3d = create_human3d_pipeline(cfg)

    print("Initialization complete, model is available")
    
# Init model
init_thread = threading.Thread(target=init_model, name="init")
init_thread.start()


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

    DataProcess.from_multipart(request, param_dict={"input_dir": upload_dir_uuid_src, "image_name": init_params["image_name"], "pcd_name": init_params["pcd_name"]})
    
    result = None

    pred_inst, pred_parts, pred_scores, full_coords = pipeline_human3d.run_pipeline(None,
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


port_id = 5001
print("API server is listening on {}".format(port_id))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port_id)
