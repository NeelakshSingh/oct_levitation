#ifndef OCT_LEVITATION_KALMAN_FILTER_H
#define OCT_LEVITATION_KALMAN_FILTER_H

#include <ros/ros.h>
#include <oct_levitation/common.h>
#include <oct_levitation/kalman_filter.h>
#include <mag_manip/forward_model_mpem.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace oct_levitation {

    class KalmanFilter {

        public:
            KalmanFilter(ros::NodeHandle& nh);

            ~KalmanFilter() = default;

        private:
            //node management
            ros::NodeHandle nh_;
            ros::Publisher state_pub_;
            ros::Subscriber vicon_sub_;

            // object properties and magnetic models
            mag_manip::DipoleVec local_dipole_moment_;
            mag_manip::ForwardModelMPEM calibration_;
            double object_mass_;
            double object_Ixxyy_;

            // state space and kalman filter related
            Eigen::MatrixXd state_space_A_;
            Eigen::MatrixXd state_space_B_;
    };

} // namespace oct_levitation

#endif // OCT_LEVITATION_KALMAN_FILTER_H