#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <vector>
#include "utils.h"

namespace experiments {
    void getTrajectoryStraightLane(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,
        size_t nb_points,
        double dist_inc,
        double x,
        double y,
        double yaw
    );

    void getTrajectoryKeepLaneFrenet(
        std::vector<double>& next_s_vals,
        std::vector<double>& next_d_vals,
        size_t nb_points,
        double s_prev,
        double s_inc,
        int lane
    );
}

#endif
