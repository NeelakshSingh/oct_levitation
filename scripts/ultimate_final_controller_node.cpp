#include <ros/ros.h>
#include <filesystem>
#include <typeinfo>
#include <Eigen/Core>

#include <mag_manip/forward_model_mpem.h>
#include <oct_levitation/common.h>


int main(int argc, char **argv) {
    ros::init(argc, argv, "ultimate_final_controller_node");
    ros::NodeHandle nh;

    // Spin the node
    ros::Rate rate(1); // 1 Hz

    std::filesystem::path calfile_path = std::filesystem::path(std::getenv("HOME") + std::string("/.ros/cal"));
    std::filesystem::path calfile_name = std::filesystem::path("mc3ao8s_md200_handp.yaml");
    calfile_path = calfile_path / calfile_name;
    
    mag_manip::ForwardModelMPEM calibration;
    calibration.setCalibrationFile(calfile_path);

    mag_manip::ActuationMat A =  calibration.getActuationMatrix(Eigen::Vector3d(0.0, 0.0, 0.0));
    ROS_INFO_STREAM("Actuation Matrix A: " << A);

    oct_levitation::RingMagnet magnet(0.0, 5e-3, 5e-3, 7500, 1.48);
    oct_levitation::RingObject body_ring(0.005, 0.025, 0.0169, 1014.9);

    oct_levitation::RingMagnetsRingObject dipole_body(
        magnet,
        0.005,
        0.025,
        0.0169,
        1014.9,
        1,
        mag_manip::DipoleVec(0.0, 0.0, -1.0),
        std::vector<double>{0.0}
    );
    
    ROS_INFO_STREAM("Magnet strength: " << magnet.getStrength());
    ROS_INFO_STREAM("Local dipole moment: " << dipole_body.getLocalDipoleMoment());
    ROS_INFO_STREAM("Object mass: " << dipole_body.mass << " Object inertia xx yy: " << dipole_body.Ixxyy);

    while (ros::ok()) {
        // Your main loop code here
        ROS_INFO("Ultimate Final Controller Node is running...");
        ROS_INFO_STREAM("Calibration file path: " << calfile_path);
        
        // Process callbacks
        ros::spinOnce();
        
        // Sleep to maintain the loop rate
        rate.sleep();
    }
    return 0;
}