cmake_minimum_required(VERSION 3.0.2)
project(oct_levitation)

set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED
              mag_manip
              control_utils
              tnb_mns_driver
              std_msgs
              geometry_msgs
              message_generation
              message_runtime
              vicon_bridge)

include_directories(${EIGEN3_INCLUDE_DIR})

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  ControllerDetails.msg
  PID1DState.msg
  Float64MultiArrayStamped.msg
  RigidBodyStateEstimate.msg
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
  xyz_reduced_attitude_single_dipole.py

  # Utilities
  scripts/utils/condition_number_plot.py
  scripts/utils/rviz_visualizations.py
  scripts/utils/coil_subset_condition_calculator.py
  scripts/utils/experiment_recorder.py
  scripts/utils/experiment_analysis.py
  scripts/utils/experiment_data_summarizer.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Installing the python libraries
catkin_python_setup()

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
)