# python infer_mhbps.py general.checkpoint='pretrained/FSK.ckpt'

import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
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
    3: (100., 143., 255.), #blue
    4: (220., 38., 127.), #pink 
    5: (255., 176., 0.), #yellow
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
        #print("EXPERIMENT ALREADY EXISTS")
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    #for log in cfg.logging:
        #print(log)

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return cfg, model, loggers


def load_ply(filepath, is_pcd=True, is_input_z_up=False, sample_max_points=200000, verbose=False):

    if is_pcd: # input is a point cloud
        pcd_input = o3d.io.read_point_cloud(filepath)
        num_total_points = len(pcd_input.points)
        num_samples = min(num_total_points, sample_max_points)
        sampled_indices = np.random.choice(num_total_points, num_samples, replace=False)
        if verbose:
            print('Total number of points:', num_total_points)
            print('Number of sampled points:', len(sampled_indices))
        pcd = copy.deepcopy(pcd_input)
        pcd = pcd.select_by_index(sampled_indices)
    else: # input is a mesh
        mesh_y_up = o3d.io.read_triangle_mesh(filepath)
        pcd_y_up_unif = mesh_y_up.sample_points_uniformly(number_of_points=sample_max_points, use_triangle_normal=False)
        pcd = copy.deepcopy(pcd_y_up_unif)

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

def process_file(filepath, scene_name='scene', is_pcd=True, is_input_z_up=False, sample_max_points=300000, verbose=False):
    coords, colors, normals = load_ply(filepath, is_pcd=is_pcd, is_input_z_up=is_input_z_up, sample_max_points=sample_max_points, verbose=verbose)

    raw_coordinates = coords.copy()
    raw_colors = colors
    raw_normals = normals

    #pdb.set_trace()
    features = np.ones(colors.shape)
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    return [[coords, features, [], scene_name, raw_colors, raw_normals, raw_coordinates, 0, None, None]], raw_coordinates


def process_pcd(pcd_coords, scene_name='scene', is_pcd=True, is_input_z_up=False, sample_max_points=300000, verbose=False):
    num_total_points = len(pcd_coords)
    if num_total_points > sample_max_points:
        num_samples = min(num_total_points, sample_max_points)
        sampled_indices = np.random.choice(num_total_points, num_samples, replace=False)
        if verbose:
            print('Total number of points:', num_total_points)
            print('Number of sampled points:', len(sampled_indices))
        pcd_coords = pcd_coords[sampled_indices, :]
   
    if not is_input_z_up:
        raise NotImplemented
        T_y_up_to_z_up = np.array([
                    [1., 0., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., -1., 0., 0.],
                    [0., 0., 0., 1.],
                ])
        pcd = pcd @ T_y_up_to_z_up[:3, :3]   

    coords = pcd_coords
    features = np.ones(pcd_coords.shape)
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    return [[coords, features, [], scene_name, np.ones(pcd_coords.shape), np.ones(pcd_coords.shape), coords, 0, None, None]], coords


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation_demo.yaml")
class Human3DPipeline():
    def __init__(self, cfg):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # because hydra wants to change dir for some reason
        os.chdir(hydra.utils.get_original_cwd())
        cfg, model, loggers = get_parameters(cfg)

        self.cfg = cfg
        self.model = model
        self.model.to(device)
        self.model.eval()

        self.c_fn = hydra.utils.instantiate(model.config.data.test_collation)
        #self.threshold = min_conf_score #0.86 #0.87 #0.89 #0.90
        #self.inst_min_point_threshold = min_num_points #8000 #2000 #for api_viz #10000
        self.iou_threshold = 0.9 #1.0


    def run_pipeline(self, pcd_coords, src_dir, res_dir, is_input_z_up=True, save_viz=False, min_conf_score=0.5, min_num_points=100):
        self.threshold = min_conf_score
        self.inst_min_point_threshold = min_num_points
        print(self.threshold, self.inst_min_point_threshold)
        if pcd_coords is None:
            filepath = os.path.join(src_dir, 'scene_pcd.ply')
            input_batch, full_coords = process_file(filepath, is_input_z_up=is_input_z_up, verbose=False)
        else:
            input_batch, full_coords = process_pcd(pcd_coords, is_input_z_up=is_input_z_up, verbose=False)
        full_coords = full_coords.astype(float)

        # process new input
        batch = self.c_fn(input_batch)

        with torch.no_grad():
            (pred_inst, pred_parts, pred_scores) = self.model.eval_step_demo(batch, threshold=self.threshold, inst_min_point_threshold=self.inst_min_point_threshold, iou_threshold=self.iou_threshold)

        # pred_inst: (num_points, ) -> instance labels, 0 is background, 1,2,... are instances
        # pred_parts: (num_points, ) -> part labels, 0 is background, 1,2,... are parts
        # full_coords: (num_points, 3)

        # save predictions for debugging
        if save_viz:
            inst_colors = (np.asarray(map2color_instances(pred_inst)).T)/255.
            part_colors = (np.asarray(map2color_parts(pred_parts)).T)/255.
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(full_coords)
            pcd.estimate_normals()
            pcd.colors = o3d.utility.Vector3dVector(inst_colors)
            o3d.io.write_point_cloud(os.path.join(res_dir, 'pred_inst.ply'), pcd)
            pcd.colors = o3d.utility.Vector3dVector(part_colors)
            o3d.io.write_point_cloud(os.path.join(res_dir, 'pred_part.ply'), pcd)
        return pred_inst.astype(np.uint8), pred_parts.astype(np.uint8), pred_scores.astype(float), full_coords
        
        
@hydra.main(config_path="conf", config_name="config_base_instance_segmentation_demo.yaml")
def create_human3d_pipeline(cfg):
    pipeline = Human3DPipeline(cfg)
    return pipeline


if __name__ == "__main__":
    create_human3d_pipeline()
