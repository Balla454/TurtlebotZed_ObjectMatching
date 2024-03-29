import os
import cv2
import pyzed.sl as sl
import pandas as pd
import numpy as np
import time
import math
import threading
import signal
import socket #added
import pickle #added

# Define the directory path and experiment name
DIR_PATH = "/home/mrrobot/Documents/ISL-Projects-main/TurtlebotZED/data_collection/10-31-23"
EXPERIMENT = 'test_1'

# Function to format the filename based on translation and rotation values
def format_filename(trans, rot):
    return f'testFiles-1_pos_{trans[0]: .2f}-{trans[1]: .2f}-{trans[2]: .2f}+rot_{rot[0]: .2f}-{rot[1]: .2f}-{rot[2]: .2f}'

# Function to initialize the camera and set its parameters
def initialize_camera():
    zed = sl.Camera()
    print("Running object detection ... Press 'Esc' to quit")
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  
    init_params.camera_fps = 60                        
    init_params.coordinate_units = sl.UNIT.FOOT
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("here1")
        print(repr(status))
        exit()
    else:
        print("here2")
    
    return zed

# Function to set runtime parameters for the ZED camera
def set_runtime_params():
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    return runtime_params

# Function to enable positional tracking on the ZED camera
def enable_positional_tracking(zed):
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_imu_fusion = True
    positional_tracking_parameters.set_as_static = False
    positional_tracking_parameters.set_floor_as_origin = True    
    zed.enable_positional_tracking(positional_tracking_parameters)

# Function to enable object detection on the ZED camera
def enable_object_detection(zed):
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

# Function to set object detection runtime parameters for the ZED camera
def set_object_detection_runtime_params():
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    detection_confidence = 60
    obj_runtime_param.detection_confidence_threshold = detection_confidence
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]
    obj_runtime_param.object_class_detection_confidence_threshold = {sl.OBJECT_CLASS.PERSON: detection_confidence} 
    return obj_runtime_param

# Function to create SDK output objects for the ZED camera
def create_sdk_output_objects(zed):
    camera_infos = zed.get_camera_information()
    point_cloud = sl.Mat(camera_infos.camera_resolution.width, camera_infos.camera_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    objects = sl.Objects()
    image_left = sl.Mat()
    display_resolution = sl.Resolution(camera_infos.camera_resolution.width, camera_infos.camera_resolution.height)
    cam_w_pose = sl.Pose()
    return point_cloud, objects, image_left, display_resolution, cam_w_pose

# Function to capture data from the ZED camera and save it to files
def capture_data(zed, runtime_params, objects, obj_runtime_param, point_cloud, image_left, display_resolution, cam_w_pose, lock):
    with lock:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            returned_state = zed.retrieve_objects(objects, obj_runtime_param)
            tracking_state = zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)    

            if (returned_state == sl.ERROR_CODE.SUCCESS and objects.is_new):
                trans = cam_w_pose.get_translation().get()
                rot = cam_w_pose.get_euler_angles()
                filename = format_filename(trans, rot)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, display_resolution)
                point_cloud.write(os.path.join(DIR_PATH, f'{filename}_pointcloud.dat'), sl.MEM.CPU) 
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_np = image_left.get_data()
                cv2.imwrite(os.path.join(DIR_PATH, f'{filename}_image.png'), image_np)
                print_zed_location(trans, rot)
                return objects, filename
    return None, None

# Function to process detected objects and return a DataFrame with their information
def process_objects(objects, cam_w_pose):
    df = pd.DataFrame(columns=['Class','Class Confidence', 'Label', 'Id', 'Object_Position', 'Object_Dimensions', '2D_Bounding_Box', '3D_Bounding_Box', 'Distance_From_Camera', 'Camera_Position'])
    print(len(objects.object_list))
    if len(objects.object_list):
        for obj in objects.object_list:
            position = obj.position
            straight = math.sqrt(position[0]**2 + position[2]**2)
            data = {'Class': obj.label, 'Class Confidence': obj.confidence,
                    'Label': obj.sublabel,
                    'Id': obj.id,
                    'Object_Position': position,
                    'Object_Dimensions': obj.dimensions,
                    '2D_Bounding_Box': obj.bounding_box_2d,
                    '3D_Bounding_Box': obj.bounding_box,
                    'Distance_From_Camera': straight,
                    'Camera_Position': cam_w_pose.pose_data()}
            df.loc[len(df)] = data
    return df

def print_zed_location(trans, rot):
    print(f"\nZED position x:{trans[0]: .2f}, y:{trans[1]: .2f}, z:{trans[2]: .2f}")
    print(f"ZED rotation: x:{rot[0]: .2f}, y:{rot[1]: .2f}, z{rot[2]: .2f}")

def update_camera(zed, runtime_parameters, stop, lock):
    while not stop[0]:
        with lock:
            zed.grab(runtime_parameters)

# Function to send the DataFrame to another device running the server script
def transmit_data(df):
    data = df.to_json()
    s = socket.socket()
    s.connect(('192.168.0.50', 16666)) # Replace 127.0.0.1 w/ the device running server script IP address
    msg = s.recv(1024)
    print(msg.decode('ascii'))
    message_to_send = 'Hello! Sending DataFrame...'
    s.send(message_to_send.encode('ascii'))
    total_sent = 0
    while total_sent < len(data):
        sent = s.send(data[total_sent:].encode())
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent
    s.close()    

# Main function that initializes the camera, sets its parameters, enables positional tracking and object detection,
# captures data, processes detected objects, and saves their information to CSV files.
def main():
    zed = initialize_camera()
    runtime_params = set_runtime_params()
    enable_positional_tracking(zed)
    enable_object_detection(zed)
    stop = [False]
    lock = threading.Lock()
    threading.Thread(target=update_camera, args=(zed, runtime_params, stop, lock)).start()
    obj_runtime_param = set_object_detection_runtime_params()
    point_cloud, objects, image_left, display_resolution, cam_w_pose = create_sdk_output_objects(zed)

    while True:
        input("Move the camera to a new location and press Enter to process data...")
        objects, filename = capture_data(zed, runtime_params, objects, obj_runtime_param, point_cloud, image_left, display_resolution, cam_w_pose, lock)
        if objects and filename:
            df = process_objects(objects, cam_w_pose)
            transmit_data(df) #sends the DataFrame to the server
            df.to_csv(os.path.join(DIR_PATH, f'data_exp_{filename}.csv'))

        quit = input("Press 'q' to quit or any other key to continue: ")
        if quit.lower() == 'q':
            stop[0] = True
            break

    image_left.free(sl.MEM.CPU)
    point_cloud.free(sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
