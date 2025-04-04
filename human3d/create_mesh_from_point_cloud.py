import open3d as o3d
from plyfile import PlyData
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse
import os


def read_plyfile(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        pcd = pd.DataFrame(plydata.elements[0].data).values
        return pcd


def create_sphere_at_xyz(xyz, colors=None, radius=0.02, resolution=4):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.compute_vertex_normals()
    if colors is None:
        sphere.paint_uniform_color([0.7, 0.1, 0.1])  # To be changed to the point color.
    else:
        sphere.paint_uniform_color(colors)
    sphere = sphere.translate(xyz)
    return sphere


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--glob_pred_paths', type=str, required=True,
                        help='glob identifier for predicted ply files')

    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to the output dir')

    parser.add_argument('--resolution', type=int, required=True,
                        help='subdivision steps (the more the better the sphere)')

    parser.add_argument('--radius', type=float, required=True,
                        help='size of sphere')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for file_id, file_path in tqdm(enumerate(sorted(list(glob(args.glob_pred_paths))))):
        pcd = read_plyfile(file_path)

        pcd_combined = o3d.geometry.TriangleMesh()

        for i in range(pcd.shape[0]):
            pcd_combined += create_sphere_at_xyz(xyz=pcd[i, :3],
                                                 colors=pcd[i, -3:] / 255.,
                                                 radius=args.radius,
                                                 resolution=args.resolution)

        o3d.io.write_triangle_mesh(f"{args.out_dir}/mesh_{file_id:05}.ply", pcd_combined)
