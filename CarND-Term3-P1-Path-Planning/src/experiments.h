#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <vector>
#include "utils.h"
#include "vehicle.h"

namespace path_planning {
    /**
     * This generates a straight line trajectory 
     **/
    void getTrajectoryStraightLane(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,
        size_t nb_points,
        double dist_inc,
        double x,
        double y,
        double yaw
    );

    /**
     * This generates a simple keep-lane trajectory (we ignore previous planned points)
     * ISSUE: speed fluctuates between 0 and input speed
     **/
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

    /**
     * This generates a keep-lane and using previous planned points
     * ISSUE: trajectory is not smooth, speed still fluctuates
     **/
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

    /**
     * This generates a keep-lane and using Spline to ensure the smooth-ness
     * ISSUE: max-acceleration and jerk is not well handled
     **/
    void getTrajectoryKeepLaneSplineConstSpeed(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const HighwayMap& highway_map,
        const Vehicle& car,
        size_t nb_points,
        double target_speed,
        int target_lane
    );

    /**
     * This generates a keep-lane using Spline
     * We ensure: max-acceleration & max-speed is handled
     *            if there is a car ahead, we should reduce speed to avoid hitting it
     **/
    void getTrajectoryKeepLaneNoRed(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              double& ref_v,
        const HighwayMap& highway_map,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion,
        size_t nb_points,
        double max_acc 
    );
}

#endif
