import argparse
import os
import pickle as pkl
import numpy as np
import smplx
from pytorch3d import transforms
from tqdm import tqdm
import sys
import pdb
from pyviz3d import visualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx_gt_dir', required=False, default=None)
    parser.add_argument('--is_input_z_up', required=True, default=None)
    parser.add_argument('--is_hi4d', action='store_true')
    args = parser.parse_args()
    smplx_gt_dir, is_input_z_up = args.smplx_gt_dir, args.is_input_z_up.lower() == "true"
    with open("results/human3d_segs/segmentation.pkl", "rb") as f:
        human3d_segs = pkl.load(f)
    segfit_inputs = {}
    for scan_name, seg in tqdm(human3d_segs.items(), file=sys.stdout):
        segfit_input = {}
        segfit_input["pcd"] = seg["points"]
        segfit_input["colors"] = seg["colors"]

        # viz = visualizer.Visualizer()
        # viz.add_points("Point Cloud", segfit_input["pcd"], 255 * seg["colors"], point_size=5)
        # viz.save("visus/" + scan_name)
        # pdb.set_trace()
        
        # divide part segmentations into one array per human instance
        processed_segs = np.zeros((np.unique(seg["instance_seg"]).shape[0]-1, seg["body_semseg"].shape[0]))
        for human_idx in np.unique(seg["instance_seg"])[1:]:
            human_colors = seg["body_semseg"].copy()
            human_colors[seg["instance_seg"] != human_idx] = 0
            processed_segs[human_idx-1] = human_colors
        # load ground truths
        if smplx_gt_dir is not None:
            # pdb.set_trace()
            fnames = [elt for elt in os.listdir(smplx_gt_dir) if scan_name.replace(".ply", "") + ".pkl" in elt]
            gt_models = []
            for fname in fnames:
                with open(os.path.join(smplx_gt_dir, fname), 'rb') as f:
                    smplx_params = pkl.load(f)
                if smplx_params["body_pose"].shape[-2] == 3:
                    smplx_params["body_pose"] = transforms.matrix_to_axis_angle(smplx_params["body_pose"]).view(-1, 63)
                if smplx_params["global_orient"].shape[-2] == 3:
                    smplx_params["global_orient"] = transforms.matrix_to_axis_angle(smplx_params["global_orient"]).unsqueeze(0)
                gt_model = smplx.create("../smplx/models/", model_type='smplx', ext="npz", gender=smplx_params.get("gender", "neutral"), use_face_contour=False, num_betas=smplx_params["betas"].shape[1])
                gt_model.reset_params(betas=smplx_params["betas"], body_pose=smplx_params["body_pose"], global_orient=smplx_params["global_orient"], transl=smplx_params["transl"])
                gt_models.append(gt_model)
        else:
            gt_models = [smplx.create("../smplx/models/", model_type='smplx', ext="npz", gender="neutral", use_face_contour=False, num_betas=10) for _ in range(max(processed_segs.shape[0], 1))]
        gt_model_outputs = [gt_model(return_verts=True, return_full_pose=True) for gt_model in gt_models]        
        rot_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) if (args.is_hi4d or not is_input_z_up) else np.eye(3)
        # save
        segfit_input["seg"] = processed_segs
        segfit_input["ground_truth_vertices"] = np.array([gt_model_output.vertices.cpu().detach().numpy().squeeze() @ rot_mat for gt_model_output in gt_model_outputs])
        segfit_input["ground_truth_joints"] = np.array([gt_model_output.joints.cpu().detach().numpy().squeeze() @ rot_mat for gt_model_output in gt_model_outputs])
        if args.is_hi4d:
            segfit_input["ground_truth_vertices"][:, :, 2] = -segfit_input["ground_truth_vertices"][:, :, 2]
            segfit_input["ground_truth_joints"][:, :, 2] = -segfit_input["ground_truth_joints"][:, :, 2]
            # segfit_input["pcd"][:, 2] = -segfit_input["pcd"][:, 2]
        segfit_inputs[scan_name] = segfit_input
    with open("results/human3d_segs/segfit_inputs.pkl", "wb") as f:
        pkl.dump(segfit_inputs, f)
    print("preparation done")