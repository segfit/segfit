import torch
import torch.nn as nn
import numpy as np
import pdb


import torch
import numpy as np

def extract_intrinsics(camera_matrix):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return fx, fy, cx, cy

def get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device="cuda:0"):
    """
    Get corresponding rays from input uv.
    i,j are flattened.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)

    rays_d = torch.sum(dirs * c2w[:3, :3], -1)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_sample_uv(H0, H1, W0, W1, depth, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..(H1-1), W0..(W1-1)

    """
    depth = depth[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device), indexing='ij')
    i = i.t()
    j = j.t()
    
    i = i.reshape(-1)
    j = j.reshape(-1)
    depth = depth.reshape(-1)

    return i, j, depth

#Define depth to distance
def depth_to_distance(depth) -> float:
    """convert depth map to a real-world distance
    """
    return -1.5 * depth + 2


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width, filter_depth_far_plane=None):
        super(BackprojectDepth, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.filter_depth_far_plane = filter_depth_far_plane

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        ).to(self.device)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False,
        ).to(self.device)

        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        ).to(self.device)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)
        self.pix_coords = self.pix_coords.double().to(self.device)

    def debugger(self, depth, inv_K):
        pdb.set_trace()

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        if self.filter_depth_far_plane is not None: #assuming that the batch size is 1
            #print('Filtering depth far plane - filtered num points:', (cam_points[0,2,:] >= self.filter_depth_far_plane).sum())
            cam_points = cam_points[:, :, cam_points[0,2,:] < self.filter_depth_far_plane]
        return cam_points  # torch.Size([1, 4, 786432])


def update_K_for_rescaling(K, h, w, h_orig, w_orig):
    scale_x = w / w_orig
    scale_y = h / h_orig
    K_rescaled = K.copy()
    K_rescaled[0, :] *= scale_x
    K_rescaled[1, :] *= scale_y
    return K_rescaled


def get_default_K():
    fx = 1597.88806
    fy = 1597.88806
    cx = 941.984863  # 0
    cy = 715.5687  # 0
    res_x = 1920
    res_y = 1440
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, res_x, res_y


def get_K_from_params(fx, fy, cx, cy, res_x, res_y):
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, res_x, res_y
