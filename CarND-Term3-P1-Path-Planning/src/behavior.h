#ifndef BEHAVIOR_H
#define BEHAVIOR_H

#include "vehicle.h"

namespace path_planning {
    
    class Behavior {
    private:
        HighwayMap _highway;
        double _look_ahead_dist;
        double _look_behind_dist;
        double _ref_v;  
        int _target_lane;
        int _step;
        
    public:
        Behavior(
            const HighwayMap& highway,
            double look_ahead_dist,
            double look_behind_dist
        );

        void reset();

        void generateTrajectory(
                std::vector<double>& next_x_vals,
                std::vector<double>& next_y_vals,
            const Vehicle& car,
            const std::vector<std::vector<double>>& sensor_fusion
        );

    };
}

#endif
