import torch
torch.cuda.empty_cache()

import pickle
import pyrender
import smplx
import open3d as o3d
import numpy as np
import os
import sys
import pdb
import time
import yaml
import argparse
import logging
from copy import deepcopy
from tqdm import tqdm
from pyviz3d import visualizer

from fitting import init_fitting, run_fitting, make_prediction, add_flipped_orientation
from utils import get_mean_error, find_matching_gt, select_humans
from visualisations import show_axes, show_segmentation, show_smplx, save_visu
from data_loading import get_scan_paths, get_smplx_paths_egobody, get_segmentations, get_smplx_paths_behave, get_smplx_paths_hi4d, get_hi4d_recording, parse_args, get_smplx_paths_poseprior
from body_part_idcs import SMPLX_2_BODY_PARTS, map2color_parts


if __name__ == "__main__":
    if not os.path.exists("results/segfit_meshes"):
        os.makedirs("results/segfit_meshes")
    config = parse_args()
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(config["data"]["log_path"], "w+"),  # Output to a log file
        ]
    )
    # read inputs
    with open("results/human3d_segs/segfit_inputs.pkl", "rb") as f:
        input_data = pickle.load(f)
    # iterate over scans
    vertex_distances, joint_distances, skipped_scans = [], [], []
    centroid_times, full_times = [], []
    iteration_count = 0
    for scan_idx, scan_name in tqdm(enumerate(sorted(list(input_data.keys()))), total=len(list(input_data.keys())), file=sys.stdout): 
        iteration_count += 1
        logging.info("\nIteration " + str(iteration_count) + ":")
        # load smplx ground truths
        ground_truth_vertices = input_data[scan_name]["ground_truth_vertices"]
        ground_truth_joints = input_data[scan_name]["ground_truth_joints"]
        # load point cloud and segmentation
        recording_points = input_data[scan_name]["pcd"]
        segmentations = input_data[scan_name]["seg"]
        if (segmentations.shape[0] == 0) or (segmentations.shape[0] == 1 and segmentations[0, :].sum() == 0):
            print("skipped scan", scan_idx, "because body part segmentation failed")
            logging.info("skipped")
            skipped_scans.append(scan_idx)
            continue
        # perform fitting
        for idx, segmentation in enumerate(segmentations):
            non_zero_idcs = np.nonzero(segmentation)[0]
            zero_idcs = np.random.choice(non_zero_idcs, size=max(0, len(non_zero_idcs)-config["fitting"]["max_points"]), replace=False)
            segmentation[zero_idcs] = 0
            segmentations[idx] = segmentation
        human_idcs = [np.nonzero(segmentation)[0] for segmentation in segmentations]
        body_parts = [segmentations[seg_idx, human_idcs[seg_idx]] for seg_idx in range(segmentations.shape[0])]
        scan_points = [torch.Tensor(recording_points[human_idcs[seg_idx]]).unsqueeze(0).to(config["fitting"]["device"]) for seg_idx in range(segmentations.shape[0])]
        orientations = [config["fitting"]["initial_orientation"]]
        orientations = add_flipped_orientation(orientations) if not config["fitting"]["use_body_parts"] else orientations
        best_loss = np.inf
        for orientation in orientations:
            model_attempts, optimizer, vposers = init_fitting(scan_points, orientation, config["fitting"]["device"], config)  
            if config["fitting"]["use_centroids"]:
                start_time = time.perf_counter()
                model_attempts, optimizer, vposers, loss = run_fitting(optimizer, config["fitting"]["max_iter"], scan_points, model_attempts, vposers, segmentations, config["fitting"]["device"], True, config["fitting"]["loss"], True, config)
                end_time = time.perf_counter()
                centroid_times.append(len(orientations) * (end_time - start_time) / segmentations.shape[0])
            if config["fitting"]["use_full_scan"]:
                start_time = time.perf_counter()
                model_attempts, optimizer, vposers, loss = run_fitting(optimizer, config["fitting"]["max_iter"], scan_points, model_attempts, vposers, segmentations, config["fitting"]["device"], config["fitting"]["use_body_parts"], config["fitting"]["loss"], False, config)
                end_time = time.perf_counter()
                full_times.append(len(orientations) * (end_time - start_time) / segmentations.shape[0])
            if loss < best_loss:
                models = model_attempts
                best_loss = loss
                model_outputs = [make_prediction(optimizer, models[j], vposers[j], config["fitting"]["device"], j) for j in range(segmentations.shape[0])]
            # compute evaluation metrics
            gt_idcs = [find_matching_gt(ground_truth_vertices, model_output.vertices) for model_output in model_outputs]
            mean_vertex_distances = [get_mean_error(model_outputs[person_idx].vertices, ground_truth_vertices[gt_idcs[person_idx]], device=config["fitting"]["device"], is_hi4d=config["data"]["is_hi4d"]) for person_idx in range(segmentations.shape[0])]
            mean_joint_distances = [get_mean_error(model_outputs[person_idx].joints, ground_truth_joints[gt_idcs[person_idx]], device=config["fitting"]["device"], is_hi4d=config["data"]["is_hi4d"], max_idx=24) for person_idx in range(segmentations.shape[0])]
            for mean_vertex_distance, mean_joint_distance in zip(mean_vertex_distances, mean_joint_distances):
                logging.info("Mean Vertex Distance: " + str(round(mean_vertex_distance, 1)) + " mm")
                logging.info("Mean Joint Distance: " + str(round(mean_joint_distance, 1)) + " mm")              
                vertex_distances.append(mean_vertex_distance)
                joint_distances.append(mean_joint_distance)          
            logging.info("Mean Centroid Fitting Time: " + str(round(centroid_times[-1], 3)) + " s") if config["fitting"]["use_centroids"] else None
            logging.info("Mean Scan Fitting Time: " + str(round(full_times[-1], 3)) + " s") if config["fitting"]["use_full_scan"] else None
            # save results
            for person_idx, model_output in enumerate(model_outputs):
                if config["visualise"]["save_meshes"]:
                    with open("results/segfit_meshes/" + scan_name.replace(".ply", "") + "_" + str(person_idx) + ".pkl", 'wb') as f:
                        pickle.dump(model_output.vertices.detach().numpy().squeeze(), f)
        # viz = visualizer.Visualizer()
        # colors = np.asarray(map2color_parts(segmentations[0])).T[segmentations[0] != 0]
        # pred_points = model_outputs[0].vertices.detach().numpy().squeeze()
        # filtered_points = recording_points[segmentations[0] != 0]
        # # viz.add_points("Point Cloud", recording_points, np.ones_like(recording_points) * np.array([100, 100, 100]), point_size=5)
        # viz.add_points("Ground Truth 1", ground_truth_vertices[0], np.ones_like(ground_truth_vertices[0]) * np.array([150, 0, 150]), point_size=5)
        # viz.add_points("Fitted Model 1", pred_points, np.ones_like(pred_points) * np.array([100, 150, 0]), point_size=5)
        # if ground_truth_vertices.shape[0] > 1:
        #     viz.add_points("Ground Truth 2", ground_truth_vertices[1], np.ones_like(ground_truth_vertices[1]) * np.array([150, 0, 150]), point_size=5)
        #     viz.add_points("Fitted Model 2", model_outputs[1].vertices.detach().numpy().squeeze(), np.ones_like(pred_points) * np.array([100, 150, 0]), point_size=5)
        #     colors = np.vstack([colors, np.asarray(map2color_parts(segmentations[1])).T[segmentations[1] != 0]])
        #     filtered_points = np.vstack([filtered_points, recording_points[segmentations[1] != 0]])
        # viz.add_points("Filtered Points", filtered_points, colors, point_size=5)
        # viz.save("visus/" + scan_name)
        # print("saved visualisation to:", "visus/" + scan_name)
        # pdb.set_trace()
    # log results
    print("number of scans skipped:", len(skipped_scans))
    logging.info("\nFINAL MEAN VERTEX DISTANCE: " + str(np.array(vertex_distances).mean().round(1)) + " mm")
    logging.info("FINAL MEAN JOINT DISTANCE: " + str(np.array(joint_distances).mean().round(1)) + " mm")
    logging.info("FINAL MEAN CENTROID FITTING TIME: " + str(np.array(centroid_times).mean().round(3)) + " s")
    logging.info("FINAL MEAN FULL FITTING TIME: " + str(np.array(full_times).mean().round(3)) + " s")
    logging.info("Processed {n} out of {m} scans".format(n=iteration_count-len(skipped_scans), m=iteration_count))
    print("logged metrics to:", config["data"]["log_path"])