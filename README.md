<div align="center">
<h1>Remote Magnetic Levitation Using Reduced Attitude Control and Parametric Field Models</h1>

<h4>Neelaksh Singh<sup>1</sup>, Jasan Zughaibi<sup>1</sup>, Denis Von Arx<sup>1</sup>, Michael M&uumlhlebach<sup>2</sup>, and Bradley J. Nelson<sup>1</sup><br></h4>
<sup>1</sup>Institute for Robotics and Intelligent Systems, ETH Z&uumlrich, Switzerland<br>
<sup>2</sup>Learning and Dynamical Systems Group, Max Planck Institute for Intelligent Systems, T&uumlbingen, Germany<br>
</div>

<h2>Code Structure</h2>
Some basic Git commands are:

```
oct_levitation
│   .gitattributes
│   .gitignore
│   CMakeLists.txt
│   debug.log
│   launch_command.txt
│   LICENSE
│   package.xml
│   README.md
│   setup.py
│
├───config
│   │   controller_params.yaml
│   │   experiment_analysis_params.yaml
│   │   general_params.yaml
│   │
│   ├───rqt_multiplot
│   │       rqt_multiplot_bronzefill_ring_27gm.xml
│   │       rqt_multiplot_currents.xml
│   │       rqt_multiplot_greentec_pro_do80_di67.xml
│   │
│   └───rviz
│           single_dipole_control_config.rviz
│
├───docs
│       rtprio_permissions.md
│
├───launch
│       control_pipeline_base.launch
│       experiment_analysis.launch
│       experiment_recording.launch
│       tnb_mns_driver.launch
│       xyz_normal_control_com_wrench.launch
│
├───msg
│       ControllerDetails.msg
│       RigidBodyStateEstimate.msg
│
├───scripts
│   │   xyz_reduced_attitude_single_dipole.py
│   │
│   └───utils
│           coil_subset_condition_calculator.py
│           condition_number_plot.py
│           experiment_analysis.py
│           experiment_data_summarizer.py
│           experiment_recorder.py
│           rviz_visualizations.py
│
└───src
    └───oct_levitation
        │   common.py
        │   control_node.py
        │   dynamics.py
        │   filters.py
        │   geometry.py
        │   geometry_jit.py
        │   mechanical.py
        │   numerical.py
        │   plotting.py
        │   processing_utils.py
        │   rigid_bodies.py
        │   trajectories.py
        │
        └───ext
                bagreader.py
                LICENSE
                __init__.py
```