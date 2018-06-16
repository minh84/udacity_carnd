#include "experiments.h"
#include "spline.h"
#include <math.h>

using namespace std;
using namespace path_planning;

namespace {
    void getTrajectoryKeepLaneFrenet(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,   
        std::vector<double>& next_s_vals,
        std::vector<double>& next_d_vals,
        const HighwayMap& highway_map,
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

        getXY(
            next_x_vals, 
            next_y_vals,
            next_s_vals,
            next_d_vals,
            highway_map
        );
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

namespace path_planning {
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
    ) {
        getTrajectoryKeepLaneFrenet(
            next_x_vals,
            next_y_vals,
            next_s_vals,
            next_d_vals,
            highway_map,
            nb_points,
            car.s(),
            s_speed, 
            target_lane
        );
    }

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
    ) {
        size_t nb_next_points = nb_points - car.prev_size();

        getTrajectoryKeepLaneFrenet(
            next_x_vals,
            next_y_vals,
            next_s_vals,
            next_d_vals,
            highway_map,
            nb_next_points,
            car.end_path_s(),
            s_speed, 
            target_lane
        );

        // append previous planned x,y to the begin of next planned x,y
        car.appendPrevPoints(next_x_vals, next_y_vals);
    }
    

    void getTrajectoryKeepLaneSplineConstSpeed(
        std::vector<double>& next_x_vals,
        std::vector<double>& next_y_vals,
        const HighwayMap& highway_map,
        const Vehicle& car,
        size_t nb_points,
        double target_speed,
        int target_lane
    ) {
        // step 1: get 2 last previous planned points to ensure Spline generate smooth trajectory
        double ref_x;
        double ref_y;
        double ref_yaw;
        vector<double> ptsx;
        vector<double> ptsy;

        car.getPrevReferencePoints(ref_x, ref_y, ref_yaw, ptsx, ptsy);
        
        // step 2: setup target points using 30m apart in Frenet (start from current position) 
        // then convert to Cartesian
        double car_s = car.s();
        std::vector<double> s_vals = {car_s + 30, car_s + 60, car_s + 90};
        std::vector<double> d_vals = std::vector<double>(3, 2 + 4 * target_lane);

        std::vector<double> x_vals;
        std::vector<double> y_vals;
        getXY(x_vals, y_vals, s_vals, d_vals, highway_map);

        // points inlcudes previous point and future points to ensure the smoothness
        ptsx.insert(ptsx.end(), x_vals.begin(), x_vals.end());
        ptsy.insert(ptsy.end(), y_vals.begin(), y_vals.end());

        // transform to local coordinate
        globalToLocal(ptsx, ptsy, ref_x, ref_y, ref_yaw);

        // spline
        tk::spline s;
        s.set_points(ptsx, ptsy);

        // plan next point
        size_t nb_next_point = nb_points - car.prev_size();
        simplePlanNextPointLocalCoord(next_x_vals, next_y_vals, s, target_speed, nb_next_point);
        localToGlobal(next_x_vals, next_y_vals, ref_x, ref_y, ref_yaw);

        // append prev points
        car.appendPrevPoints(next_x_vals, next_y_vals);
    }
}