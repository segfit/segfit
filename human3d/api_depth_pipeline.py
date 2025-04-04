import numpy as np
import os
import cv2
import torch
import open3d as o3d #0.14.1 for m1
from torchvision import transforms
from kornia.filters import MedianBlur
import pdb
from depth_utils import BackprojectDepth, update_K_for_rescaling, extract_intrinsics
import torch.nn.functional as F
import time
import copy
def get_modelzoo():
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)

    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_zoe_nk.to(device)
    model_zoe_nk.eval()
    return model_zoe_nk, device


class DepthPipeline:
    def __init__(self, filter_depth=True, filter_pcd_outliers=True, filter_depth_far_plane=None):
        self.model, self.device = get_modelzoo()
        self.rescaled_size = (192, 256) #(384, 512)
        self.backprojection_layer = BackprojectDepth(1, self.rescaled_size[0], self.rescaled_size[1], filter_depth_far_plane=filter_depth_far_plane).to(self.device)
        self.gaussian_transformer = transforms.GaussianBlur(5, 3)
        self.median_transformer = MedianBlur((5,5))
        self.filter_depth = filter_depth
        self.filter_pcd_outliers = filter_pcd_outliers
        self.T_y_up_to_z_up = np.array([ [1., 0., 0., 0.], 
                                    [0., 0.,1., 0.], 
                                    [0., -1., 0., 0.], 
                                    [0., 0., 0., 1.]]) # convert to z-up format


    def run_pipeline(self, src_dir, res_dir, write_pcd=False, image_name='image_rgb.jpg'):
        rgb_frame_path = os.path.join(src_dir, image_name)  
        K_path = os.path.join(src_dir, 'K_orig.txt')

        frame_rgb = cv2.cvtColor(cv2.imread(rgb_frame_path), cv2.COLOR_BGR2RGB) # check bgr
        K_orig = np.loadtxt(K_path)
        fx, fy, cx, cy = extract_intrinsics(K_orig)
        #print('Orig:', fx, fy, cx, cy)

        orig_size = tuple(np.asarray(frame_rgb).shape[:2])
        (orig_height, orig_width) = orig_size

        height_ratio = float(orig_size[0])/float(self.rescaled_size[0])
        width_ratio = float(orig_size[1])/float(self.rescaled_size[1])
        assert height_ratio>=1.0 and width_ratio>=1.0, "rescaled size must be smaller than original size"
        if width_ratio >= height_ratio:
            rescale_ratio = height_ratio
        else:
            rescale_ratio = width_ratio

        # K after center crop
        pre_rescale_size = (pre_rescale_height, pre_rescale_width) = (int(self.rescaled_size[0]*rescale_ratio), int(self.rescaled_size[1]*rescale_ratio)) # (height, width) -> (1080, 1440)
        cy, cx = pre_rescale_size[0]/2.0, pre_rescale_size[1]/2.0
        K_pre_rescale = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        fx, fy, cx, cy = extract_intrinsics(K_pre_rescale)

        # K after rescaling
        K_rescaled = update_K_for_rescaling(K_pre_rescale, self.rescaled_size[0], self.rescaled_size[1], pre_rescale_size[0], pre_rescale_size[1])
        fx, fy, cx, cy = extract_intrinsics(K_rescaled)
        K_inv = torch.unsqueeze(torch.linalg.inv(torch.from_numpy(K_rescaled)), dim=0).to(self.device)        

        frame_rgb = frame_rgb[orig_height-pre_rescale_height:orig_height, orig_width//2-pre_rescale_width//2:orig_width//2+pre_rescale_width//2, :]
        input_batch = transforms.functional.to_tensor(frame_rgb).unsqueeze(0).to(self.device) #torch.Size([1, 3, 1080, 1920])
        resized_input_batch = F.interpolate(input_batch, size=(self.rescaled_size[0], self.rescaled_size[1]), mode='bilinear', align_corners=False) # (1, 3, 384, 512)

        with torch.no_grad():
            prediction = self.model.infer(resized_input_batch)

        if self.filter_depth:
            prediction = self.gaussian_transformer(prediction)

        points = self.backprojection_layer(prediction, K_inv)

        points = np.squeeze(points.permute(0,2,1).cpu().numpy())[:, :3]
        
        pcd_z_up_points = points @ self.T_y_up_to_z_up[:3, :3].T
        pcd_z_up = o3d.geometry.PointCloud()
        pcd_z_up.points = o3d.utility.Vector3dVector(pcd_z_up_points)
        #pcd_z_up = pcd.transform(self.T_y_up_to_z_up)

        if self.filter_pcd_outliers:
            #pcd_z_up = pcd_z_up.voxel_down_sample(voxel_size=0.01)
            #_, ind = pcd_z_up.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.75)
            #_, ind = pcd_z_up.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
            #_, ind = pcd_z_up.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.75)
            _, ind = pcd_z_up.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.75)
            #_, ind = pcd_z_up.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            #_, ind = pcd_z_up.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            pcd_z_up = pcd_z_up.select_by_index(ind)

        if write_pcd:
            colors = cv2.resize(frame_rgb, (self.rescaled_size[1], self.rescaled_size[0]), interpolation = cv2.INTER_AREA).reshape((-1,3)) 
            pcd_z_up.colors = o3d.utility.Vector3dVector(colors/255)
            # save pcd
            o3d.io.write_point_cloud(os.path.join(src_dir, "scene_pcd.ply"), pcd_z_up)

        return np.asarray(pcd_z_up.points)

def create_depth_pipeline(filter_depth=True, filter_pcd_outliers=True, filter_depth_far_plane=None):
    pipeline = DepthPipeline(filter_depth, filter_pcd_outliers, filter_depth_far_plane)
    return pipeline
