import os
import rospkg
import rospy
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

rospy.init_node('experiment_data_summarizer_node', anonymous=True)

def node_loginfo(msg):
    rospy.loginfo(f"[oct_levitation/experiment_data_summarizer] {msg}")

def node_logerr(msg):
    rospy.logerr(f"[oct_levitation/experiment_data_summarizer] {msg}")

def node_logwarn(msg):
    rospy.logwarn(f"[oct_levitation/experiment_data_summarizer] {msg}")

rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('oct_levitation')
data_base_folder = rospy.get_param('experiment_analysis/base_folder', "")
if data_base_folder == "":
    node_loginfo("No base folder specified. Using default: oct_levitation/data/experiment_data.")
    data_base_folder = os.path.join(pkg_path, 'data', 'experiment_data')

sim = rospy.get_param('~sim') # Mandatory parameter to specify the correct folder according to experiment_recorder.py

if sim:
    data_base_folder = os.path.join(data_base_folder, 'sim')

experiment_subfolder = rospy.get_param('~data_subfolder') # mandatory parameter
data_base_folder = os.path.join(data_base_folder, experiment_subfolder)

node_loginfo(f"Using data base folder: {data_base_folder}")

# The primary purpose of this node is to collect the notes and the experiment description
# from all of the experiments and then write them into a single file for reference. For now
# nothing as advanced as copying all the plots in a specific place for comparison is planned.

# Step 1: Scan all the folders and subfolders in the data_base_folder for the observation
# notes and the experiment metadata.


def extract_description_block(desc_path):
    # Thanks ChatGPT for this function
    try:
        with open(desc_path, 'r') as f:
            lines = f.readlines()

        # Find the line where "Controller Name:" starts and keep from there
        for idx, line in enumerate(lines):
            if line.strip().startswith("Controller Name:"):
                return "".join(lines[idx:]).strip()
        return "(Ill formed experiment_description.txt)"
    except Exception as e:
        return f"(Error reading description file: {e})"

summary_file_path = os.path.join(data_base_folder, 'summary.txt')

include_description = rospy.get_param('experiment_analysis/summarizer/include_description', True)
notes_folder = rospy.get_param('experiment_analysis/summarizer/notes_folder', 'notes')

with open(summary_file_path, 'w') as summary_file:
    summary_file.write(f"Experiment data summary run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_file.write(f"Base folder: {data_base_folder}\n")
    summary_file.write(f"Experiment subfolder: {experiment_subfolder}\n")
    
    folders = [f for f in os.listdir(data_base_folder) if os.path.isdir(os.path.join(data_base_folder, f)) and f != 'summary.txt']
    summary_file.write(f"Scanning {len(folders)} folders: {folders}\n")
    folders = [os.path.join(data_base_folder, f) for f in folders]

    for folder in folders:
        summary_file.write(f"\n\n\n === {folder} ===\n")

        if include_description:
            if not os.path.exists(os.path.join(folder, 'experiment_description.txt')):
                node_logwarn(f"Experiment description file not found in {folder}. Skipping.")
                summary_file.write(f"Experiment Description: NOT FOUND\n")
            else:
                description = extract_description_block(os.path.join(folder, 'experiment_description.txt'))
                summary_file.write(f"Experiment Description: {description}\n")
        
        summary_file.write(" --- Observation Notes --- \n")

        if not os.path.exists(os.path.join(folder, 'notes')):
            summary_file.write("Observation notes folder not found.\n")
        else:
            notes_files = [f for f in os.listdir(os.path.join(folder, 'notes')) if f.endswith('.txt')]
            if len(notes_files) == 0:
                summary_file.write("No observation notes found in the notes folder.\n")
            else:
                for note_file in notes_files:
                    with open(os.path.join(folder, 'notes', note_file), 'r') as f:
                        summary_file.write(f">>> Notes from {note_file}:\n")
                        summary_file.write(f.read())
        
        summary_file.write("\n --- End of Observation Notes --- \n")
        summary_file.write("=======================================\n")