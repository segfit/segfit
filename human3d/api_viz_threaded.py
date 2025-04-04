import os
import cv2
import numpy as np
import requests
from api_viz_dicts import map2color_instances, map2color_parts
import open3d as o3d
from imutils.video import VideoStream
import time
import gzip
import orjson

url = 'http://129.132.62.174:5000/predict'  # Replace with your server's endpoint

img_write_dir = 'upload/img_write'
if not os.path.exists(img_write_dir):
    os.makedirs(img_write_dir)
image_write_path = os.path.join(img_write_dir, 'image_rgb.jpg')

# TO-DO: put your own K matrix here
K_orig_path = os.path.join(img_write_dir, 'K_orig.txt')
if not os.path.exists(K_orig_path):
    raise ValueError("Please put your own intrinsics matrix in the img_write directory")


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Human3D', visible=True)
ctr = vis.get_view_control()
render_opt = vis.get_render_option()
#render_opt.point_show_normal = True
print(render_opt.point_size)
render_opt.point_size = 10

parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera.json")
#pdb.set_trace()
#K = np.loadtxt(K_orig_path)
#parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, K[0,0], K[1,1], K[0,2], K[1,2])

print('---')

#pdb.set_trace()
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()

def producer(buffer, event):

    def keyEvent(vis):
        buffer.put(None, block=False)

    vis.register_key_callback(32, keyEvent) #spacebar

    while True:
        print("CAPTURED")
        tic = time.perf_counter()
        vis.clear_geometries()

        frame_bgr = vs.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_write_path, cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
        files = {'image_rgb':open(image_write_path, 'rb'), 'K_orig':open(K_orig_path, 'rb')}
        response = requests.post(url, files=files)

        response_dict = orjson.loads(gzip.decompress(response.content)) #dict_keys(['pred_inst', 'pred_parts', 'full_coords'])

        buffer.put(response_dict)

        tac = time.perf_counter()
        print(f"Time elapsed: {tac - tic:0.4f} seconds")
        if event.is_set():
            return

def consumer(buffer):
    while True:
        response_dict = buffer.get()

        if response_dict == None:
            print("stopped")
            return

        pred_inst = np.asarray(response_dict['pred_inst'])
        pred_parts = np.asarray(response_dict['pred_parts'])
        full_coords = np.asarray(response_dict['full_coords'])
        
        T_z_up_to_y_up = np.array([ [-1., 0., 0., 0.], 
                                    [0., 0., 1., 0.], 
                                    [0., -1., 0., 0.], 
                                    [0., 0., 0., 1.]]) 
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(full_coords)


        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
        camera_location = parameters.extrinsic[:3, 3]
        pcd.orient_normals_towards_camera_location(camera_location=camera_location)
        
        inst_colors = (np.asarray(map2color_instances(pred_inst)).T)/255.
        pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        
        # TO-DO: visualizer coordinate system should be different
        #pdb.set_trace()
        pcd = pcd.transform(T_z_up_to_y_up)
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()


    


def main():
    from threading import Thread, Event
    from queue import Queue
    
    event = Event()
    buffer = Queue()

    producer_t = Thread(target=producer, args=(buffer, event))
    producer_t.start()

    consumer_t = Thread(target=consumer, args=(buffer,))
    consumer_t.start()

    consumer_t.join()
    event.set()

    vs.stop()
    vis.destroy_window()

if __name__ == "__main__":
    main()
