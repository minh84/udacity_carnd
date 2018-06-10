#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <vector>
#include "utils.h"
#include "vehicle.h"

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

    void getTrajectorySpline(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const utils::HighwayMap& highway_map,
        const Vehicle& car,
        const std::vector<double>& prev_path_x,
        const std::vector<double>& prev_path_y,
        int target_lane,
        double target_speed
    );
}

#endif
