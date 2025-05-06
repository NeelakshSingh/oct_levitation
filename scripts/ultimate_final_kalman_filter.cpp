#include <ros/ros.h>
#include <oct_levitation/kalman_filter.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "kalman_filter_node");
    ros::NodeHandle nh;

    oct_levitation::KalmanFilter kalman_filter(nh);

    // Spin the node
    ros::Rate rate(1); // 1 Hz

    while (ros::ok()) {
        // Your main loop code here
        ROS_INFO("Kalman Filter Node is running...");

        // Sleep to maintain the loop rate
        rate.sleep();
    }
    return 0;
}