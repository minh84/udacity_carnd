#include "experiments.h"
#include <math.h>

using namespace std;
using namespace utils;

namespace experiments {
    void getTrajectoryStraightLane(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,
        size_t nb_points,
        double dist_inc,
        double x,
        double y,
        double yaw
    ) {
        // clear & reserve
        next_x_vals.resize(nb_points);
        next_y_vals.resize(nb_points);
        for (int i = 0; i < nb_points; ++i) {
            // append to trajectory
            next_x_vals[i] = x + dist_inc * i * cos(deg2rad(yaw));
            next_y_vals[i] = y + dist_inc * i * sin(deg2rad(yaw));
        }
    }

    void getTrajectoryKeepLaneFrenet(
        std::vector<double>& next_s_vals,
        std::vector<double>& next_d_vals,
        size_t nb_points,
        double s_prev,
        double s_inc,
        int lane
    ) {
        // clear & reserve
        next_s_vals.resize(nb_points);
        next_d_vals.resize(nb_points);

        double next_s = s_prev;
        double d = 2.0 + 4.0 * lane;
        for (int i = 0; i < nb_points; ++i) {
            // increase s
            next_s += s_inc;

            next_s_vals[i] = next_s;
            next_d_vals[i] = d;
        }
    }
}