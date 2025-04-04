import os
import pickle
import pdb
import numpy as np
import open3d as o3d
import torch
import yaml
import argparse

def get_scan_paths(scan_base_path, prefix=None, dataset=None):
    scan_paths = sorted(os.listdir(scan_base_path))
    if prefix is not None:
        scan_paths = [elt for elt in scan_paths if prefix in elt]
        if dataset == "hi4d":
            scan_paths = [elt for elt in scan_paths if int(elt.replace(".txt", "").split("_")[-1]) >= 100]
        elif dataset == "full_behave":
            scan_paths = [elt for elt in scan_paths if not "scenes_" in elt]
    return scan_paths

def get_smplx_paths_egobody(scan_paths, smplx_base_path):
    smplx_paths = {fname : [] for fname in scan_paths}
    for scan_path in scan_paths:
        scan = "frame".join(scan_path.replace(".ply", "").split("scene_main")) + "_000.pkl"
        for subject in ["interactee_", "camera_wearer_"]:
            smplx_paths[scan_path].append(smplx_base_path + subject + scan)
    return smplx_paths

def get_smplx_paths_behave(scan_paths, smplx_base_path):
    smplx_paths = {fname : [] for fname in scan_paths}
    for scan_path in scan_paths:
        scan = scan_path.replace("person_person", "person_fit02_person_fit").replace(".ply", ".pkl")
        smplx_paths[scan_path].append(smplx_base_path + scan)
    return smplx_paths

def get_smplx_paths_poseprior(scan_paths, smplx_base_path):
    smplx_paths = {fname : [] for fname in scan_paths}
    for scan_path in scan_paths:
        scan = scan_path.replace(".ply", ".pkl")
        smplx_paths[scan_path].append(smplx_base_path + scan)
    return smplx_paths

def get_smplx_paths_hi4d(scan_paths, smplx_base_path):
    smplx_paths = {fname : [] for fname in scan_paths}
    for scan_path in scan_paths:
        for gender in ["_female", "_male"]:
            scan = scan_path.replace(".txt", "") + gender + ".pkl"
            smplx_paths[scan_path].append(smplx_base_path + scan)
    return smplx_paths

def get_segmentations(segmentation_base_path, dataset):
    with open(segmentation_base_path + dataset + "_segmentation.pkl", "rb") as f:
        segmentations = pickle.load(f)
    return segmentations

def get_hi4d_recording(segmentations, fname, rotate=True):
    fname = fname.replace(".txt", "")
    if list(segmentations.values())[0].get("offset") is None:
        point_cloud = o3d.io.read_point_cloud("data/hi4d/human3d_training_inputs/" + fname + ".ply")
        point_offset = np.zeros(3)
        return point_cloud, point_offset
    points = []
    for key, segmentation in segmentations[fname].items():
        if key == "offset":
            continue
        points.append(segmentation[:, 0:3])
    points = np.vstack(points)
    if not rotate:
        points[:, 1] = -points[:, 1]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if rotate:
        point_offset = segmentations[fname]["offset"]
        point_offset[1], point_offset[2] = -point_offset[2], point_offset[1]
    else: 
        point_offset = None
    return point_cloud, point_offset

def parse_args():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--show")
    parser.add_argument("--use_full_scan")
    parser.add_argument("--use_centroids")
    parser.add_argument("--log_path")
    parser.add_argument("--max_points", type=int)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--use_body_parts")
    parser.add_argument("--stopping_condition", type=float)
    parser.add_argument("--save_meshes")
    parser.add_argument("--save_visu")
    parser.add_argument("--prefix")
    parser.add_argument("--fit_to_sequence")
    parser.add_argument('--is_hi4d', action='store_true')
    args = parser.parse_args()
    # load config
    base_dir = os.getcwd()
    with open(base_dir + "/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # update config
    config["data"]["base_dir"] = base_dir
    config["data"]["vposer_dir"] = base_dir + config["data"]["vposer_dir"]
    config["data"]["log_path"] = args.log_path or config["data"]["log_path"]
    config["data"]["log_path"] = os.path.join(base_dir, config["data"]["log_path"])
    config["data"]["is_hi4d"] = args.is_hi4d
    config["fitting"]["initial_orientation"] = [np.pi/2.0, 0.0, 0.0]
    config["fitting"]["min_human_dist"] = 0
    config["fitting"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["fitting"]["use_full_scan"] = (args.use_full_scan == "True") if args.use_full_scan is not None else config["fitting"]["use_full_scan"]
    config["fitting"]["use_centroids"] = (args.use_centroids == "True") if args.use_centroids is not None else config["fitting"]["use_centroids"]    
    config["fitting"]["max_points"] = args.max_points or config["fitting"]["max_points"]
    config["fitting"]["max_iter"] = args.max_iter or config["fitting"]["max_iter"]
    config["fitting"]["use_body_parts"] = (args.use_body_parts == "True") if args.use_body_parts is not None else config["fitting"]["use_body_parts"]
    config["fitting"]["use_centroids"] = False if not config["fitting"]["use_body_parts"] else config["fitting"]["use_centroids"]
    config["fitting"]["stopping_condition"] = args.stopping_condition or config["fitting"]["stopping_condition"]
    config["visualise"]["save_meshes"] = (args.save_meshes == "True") if args.save_meshes is not None else config["visualise"]["save_meshes"]
    print("Running on", config["fitting"]["device"])
    return config