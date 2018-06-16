#ifndef BEHAVIOR_H
#define BEHAVIOR_H

#include "vehicle.h"

namespace path_planning {
    /**
     * We compute the future lane using the following logic
     *      i) no car aheah: we try to drive at maximum allowed speed
     *      ii) otherwise: we try to change lane if it's safe otherwise we reduce our speed
     * The function returns
     *      target lane
     *      next_dist_adjust: the change to the distance between two steps
     */
    void planLaneAndSpeed(
              int& next_lane,
              double& next_dist_adjust,
        const Vehicle& vehicle,
        const std::vector<std::vector<double>>& sensor_fusion
    );

    /**
     * Given the lane and the distance_adjust, we generate a trajectory 
     **/
    void generateTrajectory(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const HighwayMap& highway_map,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    );
}

#endif
