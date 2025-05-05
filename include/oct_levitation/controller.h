#ifndef OCT_LEVITATION_CONTROLLER_H
#define OCT_LEVITATION_CONTROLLER_H

#include <ros/ros.h>
#include <oct_levitation/common.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <Eigen/Core>
#include <geometry_msgs/TransformStamped.h>

#include <mag_manip/forward_model_mpem.h>

namespace oct_levitation {

    class OctLevitationController {

        public:

            OctLevitationController(ros::NodeHandle& nh);
            
            ~OctLevitationController(); // default destructor for now

            mag_manip::CurrentsVec pointDipoleAllocation( 
                const mag_manip::PositionVec& position,
                const geometry_msgs::TransformStampedConstPtr& dipole_tf
            );
            
            void poseFeedbackCallback(const geometry_msgs::TransformStampedConstPtr& dipole_tf);
            
        private:

            ros::NodeHandlePtr nh_;
            std::vector<double> magnet_stack_z_positions_;
            float magnet_strength_;
            mag_manip::DipoleVec dipole_axis_;
            mag_manip::DipoleVec local_dipole_moment_;
            std::unique_ptr<ros::Subscriber> pose_feedback_sub_;
            std::unique_ptr<ros::Publisher> currents_pub_;

    }; // OctLevitationController

} // namespace oct_levitation

#endif // OCT_LEVITATION_CONTROLLER_H