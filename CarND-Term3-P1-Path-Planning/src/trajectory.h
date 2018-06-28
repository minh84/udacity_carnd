#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include "utils.h"
#include "vehicle.h"

#include <math.h>
#include <vector>

namespace path_planning {
    void getTrajectoryGivenLaneAndSpeed(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              double& ref_v,
        const HighwayMap& highway_map,
        const Vehicle& car,
        size_t nb_points,
        double max_acc,
        double target_lane,
        double target_speed
    );
}

#endif
