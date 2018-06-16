#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <vector>
#include "utils.h"
#include "vehicle.h"

namespace path_planning {
    void getTrajectoryStraightLane(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,
        size_t nb_points,
        double dist_inc,
        double x,
        double y,
        double yaw
    );

    void getTrajectoryKeepLaneV1(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              std::vector<double>& next_s_vals,
              std::vector<double>& next_d_vals,
        const HighwayMap& highway_map,
        const Vehicle& car,
        size_t nb_points,
        double s_speed,
        int target_lane
    );

    void getTrajectoryKeepLaneV2(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              std::vector<double>& next_s_vals,
              std::vector<double>& next_d_vals,
        const HighwayMap& highway_map,
        const Vehicle& car,
        size_t nb_points,
        double s_speed,
        int target_lane
    );

    void getTrajectoryKeepLaneSplineConstSpeed(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const HighwayMap& highway_map,
        const Vehicle& car,
        size_t nb_points,
        double target_speed,
        int target_lane
    );
}

#endif
