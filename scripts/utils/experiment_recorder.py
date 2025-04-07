import roslaunch.rlutil
import rospy
import roslaunch
import rospkg

import os
import sys
import signal
import subprocess

import time
from datetime import datetime

rospy.init_node('experiment_recorder_and_analysis_node', anonymous=True)


rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('oct_levitation')

# Use the current date and time stamp of the experiment as the folder to store its data
data_base_folder = os.path.join(pkg_path, 'data', 'experiment_data')
sim = rospy.get_param('~sim', False)
if sim:
    rospy.loginfo("[oct_levitation/experiment_analysis] Running in simulation mode.")
    data_base_folder = os.path.join(data_base_folder, 'sim')
    
data_folder = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
data_folder = os.path.join(data_base_folder, data_folder)
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
launchfile_parent.start()

def sigint_handler(signum, frame):
    rospy.loginfo("[oct_levitation/experiment_analysis] SIGINT received. Dumping parameter files and shutting down.")
    dump_rosparams(data_folder, 'rosparam_dump.yaml')
    launchfile_parent.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

rospy.spin() # To force SIGINT based termination through launch files.