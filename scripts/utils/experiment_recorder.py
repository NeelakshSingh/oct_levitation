import roslaunch.rlutil
import rospy
import roslaunch
import rospkg

import os
import sys
import signal
import subprocess
import shutil
import json
import ast

import time
from datetime import datetime
from oct_levitation.msg import ControllerDetails

rospy.init_node('experiment_recorder_and_analysis_node', anonymous=True)


rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('oct_levitation')

# Use the current date and time stamp of the experiment as the folder to store its data
data_base_folder = rospy.get_param('experiment_analysis/base_folder', "")
if data_base_folder == "":
    data_base_folder = os.path.join(pkg_path, 'data', 'experiment_data')
    rospy.loginfo(f"[oct_levitation/experiment_recorder] No base folder specified. Using default: {data_base_folder}.")

sim = rospy.get_param('~sim', False)
data_subfolder = rospy.get_param('~data_subfolder') # Experiment specific data_subfolder. Private and mandatory param.
if sim:
    rospy.loginfo("[oct_levitation/experiment_recorder] Running in simulation mode.")
    data_base_folder = os.path.join(data_base_folder, 'sim')
    
data_folder = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
data_folder = os.path.join(data_base_folder, data_subfolder, data_folder)


bagfile_name = 'all_topics_recording.bag'

def dump_rosparams(folder, filename):
    """
    Dump all ROS parameters to a YAML file.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, filename)
        result = subprocess.run(
            ["rosparam", "dump", file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        rospy.loginfo(f"[oct_levitation/experiment_analysis] Dumped ROS parameters to {file}.")
    except subprocess.CalledProcessError as e:
        rospy.logerr(f"[oct_levitation/experiment_analysis] rosparam dump failed to dump ROS parameters: {e}")
        rospy.logerr(e.stderr)
    except Exception as e:
        rospy.logerr(f"[oct_levitation/experiment_analysis] An error occurred while dumping ROS parameters: {e}")

os.makedirs(data_folder, exist_ok=True)

# Launchfiles for rosbag recording and experiment analysis.
experiment_recording_launch_file = os.path.join(pkg_path, 'launch', 'experiment_recording.launch')
cli_args = [
    experiment_recording_launch_file, f'folder_name:={data_folder}', f'bagfile_name:={bagfile_name}'
]
launch_args = cli_args[1:]
launch_file_format = [(
    roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], launch_args
)]
uuid = roslaunch.rlutil.get_or_generate_uuid(None, True) # Better to not let this launchfile start a roscore because of possible undefined behavior.
roslaunch.configure_logging(uuid)
launchfile_parent = roslaunch.parent.ROSLaunchParent(uuid, launch_file_format)
rospy.loginfo("[oct_levitation/experiment_analysis] Launching experiment recording node.")
launchfile_parent.start() # This call is actually non-blocking. 

def sigint_handler(signum, frame):
    rospy.loginfo("[oct_levitation/experiment_analysis] SIGINT received. Dumping parameter files and shutting down.")
    dump_rosparams(data_folder, 'rosparam_dump.yaml')
    launchfile_parent.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

# Making a backup of the controller file alongside storing its full state into the data folder.
file_backup_path = os.path.join(data_folder, 'controller_backup_files')
os.makedirs(file_backup_path, exist_ok=True)

def controller_details_callback(msg: ControllerDetails):
    """
    TODO:
    1. The message itself will be stored through the bagfile so use the details within
    to explicitly copy and paste the node running the controller into the data folder
    because that will be important in order to reproduce the experiment in the future.
    2. The reason is that the controller state may not be sufficient.
    """
    try:
        file_path = msg.controller_path.data
        controller_node_base = os.path.join(pkg_path, 'src', 'oct_levitation', 'control_node.py')
        destination_path = os.path.join(file_backup_path, os.path.basename(file_path))
        rigid_body_file = os.path.join(pkg_path, 'src', 'oct_levitation', 'rigid_bodies.py')
        shutil.copyfile(controller_node_base, os.path.join(file_backup_path, os.path.basename(controller_node_base)))
        shutil.copyfile(file_path, destination_path)
        shutil.copyfile(rigid_body_file, os.path.join(file_backup_path, os.path.basename(rigid_body_file)))
        # Saving the experiment details separately.
        desc_file = os.path.join(data_folder, 'experiment_description.txt')
        with open(desc_file, 'w') as f:
            f.write(f"Header containing launch stamp: {msg.header} \n")
            f.write(f"Controller Name: {msg.controller_name.data} \n")
            f.write(f"Controller Path: {msg.controller_path.data} \n")
            f.write(f"Data Recording Sub-Folder: {msg.data_recording_sub_folder.data} \n")
            f.write(f"Experiment Description: {msg.experiment_description.data} \n")
            f.write(f"Experiment Metadata: {msg.metadata.data} \n")

        state_file = os.path.join(data_folder, 'controller_state.json')
        with open(state_file, 'w') as f:
            json.dump(msg.full_controller_class_state.data, f, indent=4)
            
        rospy.logdebug(f"[oct_levitation/experiment_analysis] Backed up controller file to {destination_path}.")
        return True
    except Exception as e:
        rospy.logerr(f"[oct_levitation/experiment_analysis] Error while backing up controller file and data. Check if controller details are correctly filled.")
        raise

controller_details_sub = rospy.Subscriber('control_session/metadata', ControllerDetails, controller_details_callback)

rospy.spin() # To force SIGINT based termination through launch files.