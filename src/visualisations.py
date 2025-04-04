import trimesh
import pyrender
import numpy as np
from pyviz3d import visualizer
import pdb
import smplx
import os

from body_part_idcs import map2color_parts

def show_smplx(scene, model, model_output, vertex_colors=[0.3, 0.3, 0.3, 0.8], highlight_idcs=None):
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    joints = model_output.joints.detach().cpu().numpy().squeeze()
    vertex_colors = np.ones([vertices.shape[0], 4]) * vertex_colors
    if highlight_idcs is not None:
        vertex_colors[highlight_idcs] = np.ones([len(highlight_idcs), 4]) * [1.0, 0.0, 0.0, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)
    return scene

def show_axes(scene):
    active_axis = np.arange(start=-1, stop=1, step=0.01)
    other_axis = np.zeros_like(active_axis)
    axis_x = pyrender.Mesh.from_points(np.vstack([active_axis, other_axis, other_axis]).T, colors=np.vstack([np.ones_like(active_axis), other_axis, other_axis]).T) # red
    axis_y = pyrender.Mesh.from_points(np.vstack([other_axis, active_axis, other_axis]).T, colors=np.vstack([other_axis, np.ones_like(active_axis), other_axis]).T) # green
    axis_z = pyrender.Mesh.from_points(np.vstack([other_axis, other_axis, active_axis]).T, colors=np.vstack([other_axis, other_axis, np.ones_like(active_axis)]).T) # blue
    scene.add(axis_x)
    scene.add(axis_y)
    scene.add(axis_z)
    return scene

def show_segmentation(scene, body_parts, human_idcs, points):
    colours = map2color_parts(body_parts)
    colours = np.vstack([colours[0], colours[1], colours[2]]).T / 255.0
    points = points[human_idcs]
    mesh = pyrender.Mesh.from_points(points, colors=colours)
    scene.add(mesh)
    return scene

def save_visu(save_path, point_cloud=None, ground_truth=None, fitted_mesh=None, pcd_colours=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    viz = visualizer.Visualizer()
    # pdb.set_trace()
    point_cloud -= ground_truth.mean(axis=0)
    fitted_mesh -= ground_truth.mean(axis=0)
    ground_truth -= ground_truth.mean(axis=0)
    faces = smplx.create("../smplx/models/", model_type='smplx', ext="npz", gender="neutral", use_face_contour=False, num_betas=10).faces
    gt_mesh = trimesh.Trimesh(vertices=ground_truth, faces=faces, vertex_colors=np.ones_like(ground_truth) * np.array([150, 150, 150]), point_size=5)
    # pdb.set_trace()
    with open(save_path + '/../gt_mesh.ply', 'wb') as f:
        gt_mesh.export(f, file_type='ply')
    fitted_mesh = trimesh.Trimesh(vertices=fitted_mesh, faces=faces, vertex_colors=np.ones_like(fitted_mesh) * np.array([50, 50, 200]), point_size=5)
    with open(save_path + '/../fitted_mesh.ply', 'wb') as f:
        fitted_mesh.export(f, file_type='ply')
    # pdb.set_trace()
    if pcd_colours is None:
        pcd_colours = np.ones_like(point_cloud) * np.array([150, 150, 150])
    viz.add_points("Point Cloud", point_cloud, colors=pcd_colours, point_size=5)
    viz.add_points("Ground Truth", ground_truth, np.ones_like(ground_truth) * np.array([250, 0, 0]), point_size=5)
    viz.add_points("Fitted Mesh", np.asarray(fitted_mesh.vertices), np.ones_like(np.asarray(fitted_mesh.vertices)) * np.array([0, 0, 250]), point_size=5)
    # viz.add_mesh("Ground Truth", save_path + '/../gt_mesh.ply')
    # viz.add_mesh("Fitted Mesh", save_path + '/../fitted_mesh.ply')
    blender_args = {'output_path': "", 'executable_path': "", "output_prefix" : None}
    # pdb.set_trace()
    viz.save(save_path, blender_args=blender_args)