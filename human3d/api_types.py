from __future__ import annotations

import os
import csv
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from typing import Dict, List
from serde.de import deserialize
from serde.se import serialize
import json
import pdb
import numpy as np
import time
@serialize(rename_all="camelcase")
@deserialize(rename_all="camelcase")
@dataclass
class DataProcess():
    def __init__(self):
        pass

    @classmethod
    def from_multipart(cls, multipart_request, param_dict):
        image_rgb = multipart_request.files.get("image_rgb")
        if image_rgb != None:
            image_rgb.save(os.path.join(param_dict["input_dir"], param_dict["image_name"]))

        K_orig = multipart_request.files.get("K_orig")
        if K_orig != None:
            K_orig.save(os.path.join(param_dict["input_dir"], "K_orig.txt"))

        scene_pcd = multipart_request.files.get("scene_pcd")
        if scene_pcd != None:
            scene_pcd.save(os.path.join(param_dict["input_dir"], param_dict["pcd_name"]))

        #print("[INFO] Files in the src dir:", os.listdir(param_dict["input_dir"]))


class DataProcessResponse():
    def process_result(result):
        return json.dumps(result)
