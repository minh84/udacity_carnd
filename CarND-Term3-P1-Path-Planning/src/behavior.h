#ifndef BEHAVIOR_H
#define BEHAVIOR_H

#include "vehicle.h"

namespace path_planning {
    
    

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
