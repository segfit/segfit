import numpy as np
import pdb
import smplx
import pickle as pkl
import os
# from scipy.spatial import distance_matrix
import torch


def get_mean_error(pred_values, true_values, device="cpu", is_hi4d=False, max_idx=None):
    if not is_hi4d:
        mean_sq_error = pow((pred_values.detach().numpy().squeeze()[:max_idx, :] - true_values[:max_idx, :]).squeeze(), 2).mean()
    else:
        points1, points2 = torch.Tensor(pred_values.detach().numpy().squeeze()[:max_idx, :]).to(device), torch.Tensor(true_values[:max_idx, :]).to(device)
        unique_nearest_neighbors = unique_nearest_neighbor(points1, points2)
        mean_sq_error = pow(points1 - points2[unique_nearest_neighbors], 2).mean().cpu().numpy()
    return np.sqrt(mean_sq_error) * 1000 # convert to mm

def find_matching_gt(true_vertices, pred_vertices):
    distances = [pow((pred_vertices.detach().numpy().squeeze() - true_vertices[i, :]).squeeze(), 2).mean() for i in range(true_vertices.shape[0])]
    return np.argmin(distances)

def select_humans(segmentation, points, min_human_dist, min_points):
    lengths = np.zeros(segmentation.shape[0])
    means = []
    for i, instance in enumerate(segmentation):
        human_idcs = np.where(instance != 0)[0]
        means.append(points[human_idcs, :].mean(axis=0))
        lengths[i] = len(human_idcs)
    sorted_idcs = np.argsort(lengths)
    # check that returned point clouds belong to two different humans
    return_idcs = [sorted_idcs[-1]]
    for i in range(2, len(sorted_idcs)+1):
        sq_dist = np.sum(pow(means[sorted_idcs[-1]] - means[sorted_idcs[-i]], 2))
        if not np.isnan(sq_dist) and sq_dist > min_human_dist and lengths[sorted_idcs[-i]] > min_points:
            return_idcs.append(sorted_idcs[-i])
            break
    return return_idcs

def unique_nearest_neighbor(pcd1, pcd2):
    dist_matrix = torch.cdist(pcd1, pcd2)
    assignments = np.ones(pcd1.shape[0]) * -1
    for _ in range(pcd1.shape[0]):
        col_idx = dist_matrix.min(axis=0)[0].argmin().item()
        row_idx = dist_matrix[:, col_idx].argmin().item()
        assignments[row_idx] = col_idx
        dist_matrix[row_idx, :] = float('inf')
        dist_matrix[:, col_idx] = float('inf')
    return torch.tensor(assignments, dtype=torch.long, device=pcd1.device)
