#include "experiments.h"
#include "spline.h"
#include <math.h>

using namespace std;
using namespace utils;

namespace {
    void getPrevReferencePoints(
              double& ref_x,
              double& ref_y,
              double& ref_yaw,
              std::vector<double>& ptsx,
              std::vector<double>& ptsy,
        const Vehicle& car,
        const std::vector<double>& prev_path_x,
        const std::vector<double>& prev_path_y
    ) {
        // we use previous planned points as reference
        size_t prev_size = prev_path_x.size();

        double prev_x, prev_y;

        if (prev_size < 2) {
            ref_x = car.x();
            ref_y = car.y();
            ref_yaw = utils::deg2rad(car.yaw());
            prev_x = ref_x - cos(ref_yaw);
            prev_y = ref_y - sin(ref_yaw);
        } else {
            ref_x = prev_path_x[prev_size - 1];
            ref_y = prev_path_y[prev_size - 1];
            prev_x = prev_path_x[prev_size - 2];
            prev_y = prev_path_y[prev_size - 2];
            ref_yaw = atan2(ref_y - prev_y, ref_x - prev_x);
        }

        // use 2 previous points to ensure smoothness since Spline is continuous in values and derivatives
        ptsx.push_back(prev_x); 
        ptsx.push_back(ref_x);

        ptsy.push_back(prev_y); 
        ptsy.push_back(ref_y);
    }

    // this plan next points using Spline in local coordinate with const speed
    void simplePlanNextPointLocalCoord(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const tk::spline& s,
        double target_speed,
        size_t nb_points
    ) {
         // calculate point in 30m ahead (using car's local coordinate)
        double local_target_x = 30;
        double local_target_y = s(local_target_x);
        double target_dist = sqrt(local_target_x * local_target_x + local_target_y * local_target_y);

        // use constant step
        double N = target_dist/target_speed;
        double x_step = local_target_x / N;

        next_x_vals.resize(nb_points);
        next_y_vals.resize(nb_points);
        double x = x_step;
        for(int i = 0; i < nb_points; ++i) {
            next_x_vals[i] = x;
            next_y_vals[i] = s(x);
            x += x_step;
        }
    }
}

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

    void getTrajectorySpline(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,
        const utils::HighwayMap& highway_map,
        const Vehicle& car,
        const std::vector<double>& prev_path_x,
        const std::vector<double>& prev_path_y,
        int target_lane,
        double target_speed
    ) {
        double ref_x;
        double ref_y;
        double ref_yaw;
        vector<double> ptsx;
        vector<double> ptsy;

        getPrevReferencePoints(ref_x, ref_y, ref_yaw, ptsx, ptsy, car, prev_path_x, prev_path_y);
        
        // setup target points using 30m apart in Frenet then convert to Cartesian
        double car_s = car.s();
        std::vector<double> s_vals = {car_s + 30, car_s + 60, car_s + 90};
        std::vector<double> d_vals = std::vector<double>(3, 2 + 4 * target_lane);

        std::vector<double> x_vals;
        std::vector<double> y_vals;
        utils::getXY(x_vals, y_vals, s_vals, d_vals, highway_map);

        // points inlcudes previous point and future points to ensure the smoothness
        ptsx.insert(ptsx.end(), x_vals.begin(), x_vals.end());
        ptsy.insert(ptsy.end(), y_vals.begin(), y_vals.end());

        // transform to local coordinate
        utils::globalToLocal(ptsx, ptsy, ref_x, ref_y, ref_yaw);

        // spline
        tk::spline s;
        s.set_points(ptsx, ptsy);

        // plan next point
        size_t nb_next_point = 50 - prev_path_x.size();
        simplePlanNextPointLocalCoord(next_x_vals, next_y_vals, s, target_speed, nb_next_point);
        utils::localToGlobal(next_x_vals, next_y_vals, ref_x, ref_y, ref_yaw);
    }
}