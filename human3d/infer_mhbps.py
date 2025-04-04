# python infer_mhbps.py general.checkpoint='pretrained/FSK.ckpt'

import os
import sys
import logging
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything
import open3d as o3d
import numpy as np
import torch
import copy
import time
import pdb
import pickle
from tqdm import tqdm
from pyviz3d import visualizer


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

LABEL_MAP = {
    0: "background",
    1: "rightHand",
    2: "rightUpLeg",
    3: "leftArm",
    4: "head",
    5: "leftLeg",
    6: "leftFoot",
    7: "torso",
    8: "rightFoot",
    9: "rightArm",
    10: "leftHand",
    11: "rightLeg",
    12: "leftForeArm",
    13: "rightForeArm",
    14: "leftUpLeg",
    15: "hips",
}

COLOR_MAP_INSTANCES = {
    0: (226., 226., 226.), #(174., 199., 232.),
    1: (120., 94., 240.), #purple 
    2: (254., 97., 0.), #orange
    3: (255., 176., 0.), #yellow
    4: (100., 143., 255.), #blue
    5: (220., 38., 127.), #pink
    6: (0., 255., 255.),
    7: (255., 204., 153.),
    8: (255., 102., 0.),
    9: (0., 128., 128.),
    10: (153., 153., 255.),
}

COLOR_MAP_PARTS = {
    0:  (226., 226., 226.),
    1:  (158.0, 143.0, 20.0),  #rightHand
    2:  (243.0, 115.0, 68.0),  #rightUpLeg
    3:  (228.0, 162.0, 227.0), #leftArm
    4:  (210.0, 78.0, 142.0),  #head
    5:  (152.0, 78.0, 163.0),  #leftLeg
    6:  (76.0, 134.0, 26.0),   #leftFoot
    7:  (100.0, 143.0, 255.0), #torso
    8:  (129.0, 0.0, 50.0),    #rightFoot
    9:  (255., 176., 0.),      #rightArm
    10: (192.0, 100.0, 119.0), #leftHand
    11: (149.0, 192.0, 228.0), #rightLeg 
    12: (243.0, 232.0, 88.0),  #leftForeArm
    13: (90., 64., 210.),      #rightForeArm
    14: (152.0, 200.0, 156.0), #leftUpLeg
    15: (129.0, 103.0, 106.0), #hips
}

map2color_instances = np.vectorize({key: item for key, item in COLOR_MAP_INSTANCES.items()}.get)
map2color_parts = np.vectorize({key: item for key, item in COLOR_MAP_PARTS.items()}.get)


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    cfg.general.seed = 444444
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return cfg, model, loggers


def load_ply(filepath, is_pcd=True, is_input_z_up=False, sample_max_points=200000, verbose=False):

    pcd = o3d.io.read_point_cloud(filepath)

    if not is_input_z_up:
        T_y_up_to_z_up = np.array([
                    [1., 0., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., -1., 0., 0.],
                    [0., 0., 0., 1.],
                ])
        pcd = pcd.transform(T_y_up_to_z_up)        

    pcd.estimate_normals()
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    return coords, colors, normals

def process_file(filepath, scene_name, is_pcd=True, is_input_z_up=False, sample_max_points=1000000, verbose=False):
    coords, colors, normals = load_ply(filepath, is_pcd=is_pcd, is_input_z_up=is_input_z_up, sample_max_points=sample_max_points, verbose=verbose)

    raw_coordinates = coords.copy()
    raw_colors = colors
    raw_normals = normals

    if is_input_z_up and "hi4d_separated" in filepath:
        coords[:, 2] = -coords[:, 2] # NOTE: flipping z axis

    features = np.ones_like(coords)
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    return [[coords, features, [], scene_name, raw_colors, raw_normals, raw_coordinates, 0, None, None]], raw_coordinates




@hydra.main(config_path="conf", config_name="config_base_instance_segmentation_demo.yaml")
def test(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    full_results = {}

    scenes_base_path = cfg.segfit.data_path
    fnames = sorted(os.listdir(scenes_base_path))
    for fname in tqdm(fnames, file=sys.stdout):
        if not fname.endswith(".ply"):
            continue
        filepath = scenes_base_path + fname

        # because hydra wants to change dir for some reason
        os.chdir(hydra.utils.get_original_cwd())
        cfg, model, loggers = get_parameters(cfg)
        model.to(device)
        model.eval()

        c_fn = hydra.utils.instantiate(model.config.data.test_collation)


        # Example scenes
        scene_name = filepath.split("/")[-1].replace('.ply', "")
        # is_input_z_up = True #: z-up, False: y-up

        # process new input
        input_batch, full_coords = process_file(filepath, scene_name=scene_name, is_input_z_up=cfg.segfit.is_input_z_up, verbose=False)
        batch = c_fn(input_batch)
        with torch.no_grad():
            (pred_inst, pred_parts) = model.eval_step_demo(batch)[:2]

        # # save predictions for debugging
        # inst_colors = (np.asarray(map2color_instances(pred_inst)).T)/255.
        part_colors = (np.asarray(map2color_parts(pred_parts)).T)/255.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(full_coords)
        pcd.estimate_normals()
        pcd.colors = o3d.utility.Vector3dVector(part_colors)
        print("Number of detected human points:", sum(pred_parts != 0))

        # viz = visualizer.Visualizer()
        # viz.add_points("Point Cloud", full_coords, 255 * part_colors, point_size=5)
        # viz.save("visus/" + scene_name)
        # pdb.set_trace()

        # o3d.io.write_point_cloud(f'results/human3d_segs/{scene_name}.ply', pcd)
        result = {"body_semseg" : pred_parts, "instance_seg" : pred_inst, "points" : np.asarray(pcd.points), "colors" : np.asarray(pcd.colors)}
        full_results[scene_name] = result
    with open("results/human3d_segs/segmentation.pkl", "wb") as f:
        pickle.dump(full_results, f)
    
    
@hydra.main(config_path="conf", config_name="config_base_instance_segmentation_demo.yaml")
def main(cfg: DictConfig):
    if not os.path.exists("../../../../results/human3d_segs"):
        os.makedirs("../../../../results/human3d_segs")
    test(cfg)
    """Run with:
    python infer_mhbps.py general.checkpoint='pretrained/FSK.ckpt'
    """


if __name__ == "__main__":
    main()
