import torch
import smplx
import numpy as np
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
from torch.optim import Adam
import pyrender
import trimesh
from scipy.spatial.transform import Rotation
from pytorch3d import transforms
import pdb
import open3d as o3d
import time

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from body_part_idcs import BODY_PART_IDCS
from visualisations import show_smplx, show_axes, show_segmentation
from chamfer_adaptation import chamfer_distance


def make_prediction(optimizer, model, vposer, device, instance_idx):
    params = list(optimizer.param_groups[0]["params"])[4*instance_idx:4*(instance_idx+1)]
    vposer_representation, global_orientation, global_translation, shape_params = params
    smplx_representation = vposer.decode(vposer_representation.to("cpu"))["pose_body"].contiguous().view(-1, 63).to(device)
    model_output = model(return_verts=True, return_full_pose=True, body_pose=smplx_representation.to("cpu"), global_orient=global_orientation.to("cpu"), transl=global_translation.to("cpu"), betas=shape_params.to("cpu"))
    return model_output

def get_chamfer_loss(optimizer, model, vposer, scan_tensor, segmentation, device, use_body_parts, loss_type, use_only_means, config, instance_idx):
    model_output = make_prediction(optimizer, model, vposer, device, instance_idx)
    model_vertices = model_output.vertices.to(device)
    human_seg = segmentation[np.where(segmentation != 0)[0]]
    chamfer_loss = 0
    if use_body_parts:
        for body_part, vertex_idcs in BODY_PART_IDCS.items():
            body_part_idcs = np.where(human_seg == body_part)[0]
            if len(body_part_idcs) == 0:
                continue
            body_part_vertices = model_vertices[:, vertex_idcs, :]
            body_part_points = scan_tensor[:, body_part_idcs, :]
            if use_only_means:
                body_part_vertices = body_part_vertices.mean(axis=1, keepdim=True)
                body_part_points = body_part_points.mean(axis=1, keepdim=True)
            chamfer_point_dist, _ = chamfer_distance(body_part_points, body_part_vertices, x_normals=None, y_normals=None, point_reduction="sum", batch_reduction=None, abs_cosine=False, custom_norm=loss_type, single_directional=True)
            chamfer_loss += chamfer_point_dist
    else:
        chamfer_point_dist, _ = chamfer_distance(scan_tensor, model_vertices, x_normals=None, y_normals=None, point_reduction="sum", batch_reduction=None, abs_cosine=False, custom_norm=loss_type, single_directional=True)
        chamfer_loss += chamfer_point_dist
    if use_only_means:
        return config["weights"]["chamfer_centroid"] * chamfer_loss.squeeze() / len(BODY_PART_IDCS.keys())
    else:
        return config["weights"]["chamfer_full"] * chamfer_loss.squeeze() / len(human_seg)

def get_penetration_loss(use_only_means, current_model, frozen_model, device, max_points=100000):
    if use_only_means:
        return 0.0
    # get model vertices and normals
    model_output1 = current_model(return_verts=True, return_full_pose=True)
    vertices1 = model_output1.vertices.squeeze().to(device)
    faces1 = current_model.faces
    mesh1 = trimesh.Trimesh(vertices=vertices1.cpu().detach().numpy(), faces=faces1)
    vertex_normals1 = mesh1.vertex_normals
    model_output2 = frozen_model(return_verts=True, return_full_pose=True)
    vertices2 = model_output2.vertices.squeeze().to(device)
    faces2 = frozen_model.faces
    mesh2 = trimesh.Trimesh(vertices=vertices2.cpu().detach().numpy(), faces=faces2)
    vertex_normals2 = mesh2.vertex_normals
    # subsample points for time efficiency
    subset_idcs = np.random.choice(vertices1.shape[0], size=min(max_points, vertices1.shape[0]), replace=False)
    vertices1 = vertices1[subset_idcs]
    vertices2 = vertices2[subset_idcs]
    vertex_normals1 = vertex_normals1[subset_idcs]
    vertex_normals2 = vertex_normals2[subset_idcs]
    # compute distances
    num_points = vertices1.shape[0]
    vertices1_expanded = vertices1.unsqueeze(0).expand(num_points, num_points, 3)
    vertices2_expanded = vertices2.unsqueeze(1).expand(num_points, num_points, 3)
    distances = torch.norm(vertices1_expanded - vertices2_expanded, dim=2)
    min_distances, closest_idcs = torch.min(distances, dim=0)
    # compute signs of distances
    closest_points = vertices2[closest_idcs]
    vectors_to_closest_points = closest_points - vertices1
    normals_at_closest_points = torch.from_numpy(vertex_normals2[closest_idcs.to("cpu")]).to(device)
    dot_products = torch.sum(vectors_to_closest_points * normals_at_closest_points, dim=1)
    signed_distances = torch.where(dot_products < 0, min_distances, -min_distances)
    # compute loss
    penetration_mask = signed_distances < 0
    loss = signed_distances[penetration_mask].sum().abs()
    # repeat in reversed direction
    min_distances_rev, closest_idcs_rev = torch.min(distances, dim=1)
    closest_points_rev = vertices1[closest_idcs_rev]
    vectors_to_closest_points_rev = closest_points_rev - vertices2
    normals_at_closest_points_rev = torch.from_numpy(vertex_normals1[closest_idcs_rev.to("cpu")]).to(device)
    dot_products_rev = torch.sum(vectors_to_closest_points_rev * normals_at_closest_points_rev, dim=1)
    signed_distances_rev = torch.where(dot_products_rev < 0, min_distances_rev, -min_distances_rev)
    penetration_mask_rev = signed_distances_rev < 0
    loss += signed_distances_rev[penetration_mask_rev].sum().abs()
    return loss

def compute_loss(optimizer, scan_tensor, models, vposer, segmentation, device, use_body_parts, loss_type, use_only_means, config, instance_idx):
    model = models[instance_idx]
    params = list(optimizer.param_groups[0]["params"])[4*instance_idx:4*(instance_idx+1)]
    # penalize deviation from pose prior
    vposer_representation = params[0]
    loss = config["weights"]["pose_prior"] * vposer_representation.pow(2).sum()
    shape_params = params[3]
    loss += config["weights"]["shape_prior"] * shape_params.pow(2).sum()
    # penalize deviation from target point cloud
    loss += get_chamfer_loss(optimizer, model, vposer, scan_tensor, segmentation, device, use_body_parts, loss_type, use_only_means, config, instance_idx)
    return loss

def run_fitting(optimizer, max_iters, scan_tensors, models, vposers, segmentations, device, use_body_parts, loss_type, use_only_means, config):
    prev_losses = [np.inf for _ in range(5)]
    loss = torch.Tensor([0.0])
    for i in tqdm(range(max_iters)):
        optimizer.zero_grad()
        loss = 0.0
        for j in range(len(models)):
            loss += compute_loss(optimizer, scan_tensors[j], models, vposers[j], segmentations[j, :], device, use_body_parts, loss_type, use_only_means, config, j)
        loss.backward()
        optimizer.step()
        if abs(loss.item() - np.mean(prev_losses)) < config["fitting"]["stopping_condition"]:
            break
        prev_losses.pop(0)
        prev_losses.append(loss.item())
    return models, optimizer, vposers, loss.item()

def init_fitting(scan_tensors, orientation, device, config):
    parameters = []
    models = []
    vposers = []
    for scan_tensor in scan_tensors:
        # initialize model parameters
        global_translation = scan_tensor.mean(axis=1) + torch.Tensor(config["fitting"]["initial_translation"]).to(device)
        global_translation.requires_grad_(True)
        global_orientation = torch.tensor([orientation]).to(device)
        global_orientation.requires_grad_(True)
        shape_params = torch.zeros(1, 10).to(device)
        shape_params.requires_grad_(True)
        pose_params = torch.zeros(1, 32)
        # initialize vposer
        vposer, _ = load_model(config["data"]["vposer_dir"], model_code=VPoser, disable_grad=False, remove_words_in_model_weights='vp_model.')
        vposer_representation = pose_params.to(device)
        vposer_representation.requires_grad_(True)
        # initialize smplx
        smplx_representation = vposer.decode(vposer_representation.to("cpu"))["pose_body"].contiguous().view(-1, 63).to(device)
        model = smplx.create("../smplx/models/", model_type='smplx', ext="npz", gender="male", use_face_contour=False, num_betas=10, device=device)
        model.reset_params(betas=shape_params, body_pose=smplx_representation, global_orient=global_orientation, transl=global_translation)
        # gather results
        models.append(model)
        vposers.append(vposer)
        [parameters.append(parameter) for parameter in [vposer_representation, global_orientation, global_translation, shape_params]]            
    optimizer = Adam(parameters, lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)
    return models, optimizer, vposers

def add_flipped_orientation(orientations):
    prior_orientation = torch.tensor(orientations[0], dtype=torch.float64)
    prior_rot_matrix = transforms.axis_angle_to_matrix(prior_orientation)
    # rotate about y axis
    for angle in [np.pi/2.0, np.pi, 3.0*np.pi/2.0 ]:    
        flip_mat = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        flipped_rot_mat = flip_mat @ prior_rot_matrix
        flipped_orientation = transforms.matrix_to_axis_angle(flipped_rot_mat)
        orientations.append(flipped_orientation.tolist())
    return orientations
