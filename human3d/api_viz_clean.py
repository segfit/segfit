import os
import cv2
import numpy as np
import requests
import json
import pdb
from api_viz_dicts import map2color_instances, map2color_parts
import open3d as o3d
from imutils.video import VideoStream
import time
import threading
import gzip
import orjson
import PIL.Image as Image
from depth_utils import update_K_for_rescaling, extract_intrinsics


url = 'http://129.132.62.174:5000/predict'
img_write_dir = 'upload/img_write'

if not os.path.exists(img_write_dir):
    os.makedirs(img_write_dir)
image_write_path = os.path.join(img_write_dir, 'image_rgb.jpg')
image_write_resized_path = os.path.join(img_write_dir, 'image_rgb_resized.jpg')

# TO-DO: put your own K matrix here
K_orig_path = os.path.join(img_write_dir, 'K_orig.txt')
K_orig_resized_path = os.path.join(img_write_dir, 'K_orig_resized.txt')
if not os.path.exists(K_orig_path):
    raise ValueError("Please put your own intrinsics matrix in the img_write directory")

suppress_overlapping_masks = True
T_z_up_to_y_up = np.array([ [-1., 0., 0., 0.], 
                            [0., 0., 1., 0.], 
                            [0., -1., 0., 0.], 
                            [0., 0., 0., 1.]]) 

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Human3D', visible=True)
ctr = vis.get_view_control()
render_opt = vis.get_render_option()
render_opt.point_size = 10
show_inst = False #show instances if True, show body parts if False
parameters = o3d.io.read_pinhole_camera_parameters("ScreenCameraDemo.json")
#K = np.loadtxt(K_orig_path)
#parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, K[0,0], K[1,1], K[0,2], K[1,2])

print('---')

#pdb.set_trace()
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
response = None
prev_response = None
files = None
pcd = o3d.geometry.PointCloud() # None
frame_bgr = vs.read()
key = cv2.waitKey(1000) & 0xFF

counter = 0

def continuously_capture_send_request_get_response(image_write_path, K_orig_path, resize_img=True, rescaled_size=(384, 512)):
    while True:
        print("CAPTURED")
        global vs, files, vis_data_unpacked, response, response_received_event, normals_ready_event
        start = time.perf_counter()
        t1 = time.perf_counter()
        frame_bgr = vs.read()
        t2 = time.perf_counter()
        if resize_img:
            K_orig = np.loadtxt(K_orig_path)
            fx, fy, cx, cy = extract_intrinsics(K_orig)
            orig_size = tuple(np.asarray(frame_bgr).shape[:2])
            (orig_height, orig_width) = orig_size

            height_ratio = float(orig_size[0])/float(rescaled_size[0])
            width_ratio = float(orig_size[1])/float(rescaled_size[1])
            assert height_ratio>=1.0 and width_ratio>=1.0, "rescaled size must be smaller than original size"
            if width_ratio >= height_ratio:
                rescale_ratio = height_ratio
            else:
                rescale_ratio = width_ratio

            # K after center crop
            pre_rescale_size = (pre_rescale_height, pre_rescale_width) = (int(rescaled_size[0]*rescale_ratio), int(rescaled_size[1]*rescale_ratio)) # (height, width) -> (1080, 1440)
            cy, cx = pre_rescale_size[0]/2.0, pre_rescale_size[1]/2.0
            K_pre_rescale = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            fx, fy, cx, cy = extract_intrinsics(K_pre_rescale)

            # K after rescaling
            K_rescaled = update_K_for_rescaling(K_pre_rescale, rescaled_size[0], rescaled_size[1], pre_rescale_size[0], pre_rescale_size[1])
            fx, fy, cx, cy = extract_intrinsics(K_rescaled)

            frame_bgr = frame_bgr[orig_height-pre_rescale_height:orig_height, orig_width//2-pre_rescale_width//2:orig_width//2+pre_rescale_width//2, :]
            frame_bgr = cv2.resize(frame_bgr, (rescaled_size[1], rescaled_size[0]))

            cv2.imwrite(image_write_resized_path, frame_bgr)
            np.savetxt(K_orig_resized_path, K_rescaled)

            t3 = time.perf_counter()
            files = {'image_rgb':open(image_write_resized_path, 'rb'), 'K_orig':open(K_orig_resized_path, 'rb')}
            t4 = time.perf_counter()

        else:
            cv2.imwrite(image_write_path, frame_bgr)
            t3 = time.perf_counter()
            files = {'image_rgb':open(image_write_path, 'rb'), 'K_orig':open(K_orig_path, 'rb')}
            t4 = time.perf_counter()

        
        response = requests.post(url, files=files)
        t5 = time.perf_counter()

        response_received_event.set()
        normals_ready_event.set()
        end = time.perf_counter()
        print('send request get response', t5-t4)

        print('capture_send_request_get_response time elapsed: ', end-start)

def estimate_normals():
    start = time.perf_counter()
    global pcd, normals_ready_event
    if normals_ready_event.is_set():
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=20))
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))
        #pcd.estimate_normals()
        normals_ready_event.clear()
    end = time.perf_counter()
    #print('-----estimate_normals time elapsed: ', end-start)
    return


def process_response_update_vis(show_inst=True):
    global vis, ctr, parameters, T_z_up_to_y_up, response, prev_response, pcd, response_received_event, request_ready_event, vis_data_unpacked, normals_ready_event
    #print('response received:', response_received_event.is_set())
    #response_received_event.wait()
    #response_received_event = threading.Event()
    #start = time.perf_counter()
    if prev_response == response:
        return

    response_dict = orjson.loads(gzip.decompress(response.content)) #dict_keys(['pred_inst', 'pred_parts', 'full_coords'])
    prev_response = response
    vis_data_unpacked.set()

    t0 = time.perf_counter()
    full_coords = np.asarray(response_dict['full_coords'])
    #pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_coords)
    pcd = pcd.transform(T_z_up_to_y_up)
    
    #pred_scores = np.asarray(response_dict['pred_scores'])

    if show_inst:
        pred_inst = np.asarray(response_dict['pred_inst'])
        inst_idx_list = np.unique(pred_inst)
        inst_remapper = {el:el for el in inst_idx_list}

        #"""
        # sort from left to right
        if len(inst_idx_list) > 1:
            inst_centers = np.asarray([np.mean(full_coords[pred_inst==idx], axis=0) for idx in inst_idx_list if idx!=0]) # indices:inst_idx-1
            res = np.argsort(-inst_centers[:, 0]) # increasing order
            for orig_inst_idx in inst_idx_list:
                if orig_inst_idx != 0:
                    inst_remapper[orig_inst_idx] = res[orig_inst_idx-1] + 1
            inst_remapper[0] = 0
            #print(inst_remapper)
        #"""

        inst_remapper_vectorized = np.vectorize({key: item for key, item in inst_remapper.items()}.get)

        inst_colors = (np.asarray(map2color_instances(inst_remapper_vectorized(pred_inst)), dtype=np.float64).T)/255.
        pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        #o3d.io.write_point_cloud(os.path.join(img_write_dir, 'pred_inst.ply'), pcd)

    else:
        pred_parts = np.asarray(response_dict['pred_parts'])
        part_colors = (np.asarray(map2color_parts(pred_parts)).T)/255.
        pcd.colors = o3d.utility.Vector3dVector(part_colors)
        #o3d.io.write_point_cloud(os.path.join(img_write_dir, 'pred_part.ply'), pcd)
    
    estimate_normals()
    #end = time.perf_counter()
    #print('process_response_update_vis time elapsed: ', end-start)

    #normals_ready_event.set()
    return #pcd


request_ready_event = threading.Event()
response_received_event = threading.Event()
vis_data_unpacked = threading.Event()
vis_data_unpacked.set()
normals_ready_event = threading.Event()


initial = True

thread_capture_send_get_response = threading.Thread(target=continuously_capture_send_request_get_response, args=(image_write_path, K_orig_path))
thread_capture_send_get_response.start()
response_received_event.wait()


while True:
    #tic = time.perf_counter()

    thread_visualize_results = threading.Thread(target=process_response_update_vis, args=(show_inst,))
    thread_visualize_results.start()
    thread_visualize_results.join()

    vis.clear_geometries()
    vis.add_geometry(pcd)
    #vis.update_geometry(pcd)

    ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()

    key = cv2.waitKey(5) & 0xFF
    
    if key == ord("q"):
        break

    #tac_final = time.perf_counter()
    #print(f"Iter time elapsed: {tac_final - tic:0.4f} seconds")

key = cv2.waitKey(30) & 0xFF
cv2.destroyAllWindows()
vs.stop()
#vis.destroy_window()
