#ifndef OCT_LEVITATION_KALMAN_FILTER_H
#define OCT_LEVITATION_KALMAN_FILTER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace oct_levitation {

    class KalmanFilter {

        public:
            KalmanFilter();

            ~KalmanFilter();

        private:
            Eigen::MatrixXd state_space_A_;
            Eigen::MatrixXd state_space_B_;
    };

} // namespace oct_levitation

#endif // OCT_LEVITATION_KALMAN_FILTER_H