#include "behavior.h"
#include "spline.h"
#include <math.h>

using namespace path_planning;
using namespace std;

namespace {
    void splinePlanNextPoint(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const tk::spline& s,
        double ref_dist,
        double dist_adjust,
        size_t nb_points
    ) {
         // calculate point in 30m ahead (using car's local coordinate)
        double local_target_x = 30;
        double local_target_y = s(local_target_x);
        double target_dist = sqrt(local_target_x * local_target_x + local_target_y * local_target_y);

        next_x_vals.resize(nb_points);
        next_y_vals.resize(nb_points);

        double cur_dist = ref_dist;
        double cur_x = 0.;
        for(int i = 0; i < nb_points; ++i) {
            cur_dist += dist_adjust;
            cur_dist = capDist(cur_dist);

            double fraction = cur_dist / target_dist;
            cur_x += fraction * local_target_x;

            next_x_vals[i] = cur_x;
            next_y_vals[i] = s(cur_x);
        }
    }
}

namespace path_planning {
    void planLaneAndSpeed(
              int& next_lane,
              double& next_dist_adjust,
        const Vehicle& vehicle,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        int current_lane = vehicle.lane();

        // check if any surrounding car is closed by one the left, 
        // ahead and the right of our vehicle
        bool car_left = false;
        bool car_ahead = false;
        bool car_right = false;

        for(size_t i = 0; i < sensor_fusion.size(); ++i) {
            int car_i_lane;
            double car_i_future_s;

            vehicle.getFuturePosition(
                car_i_lane, 
                car_i_future_s,
                sensor_fusion[i]
            );
            
            if (car_i_lane==-1) { // ignore car not in the same part of the road
                continue;
            }

            if (car_i_lane == current_lane) {
                car_ahead |= vehicle.isAhead(car_i_future_s);
            } else if (car_i_lane == current_lane - 1) { // car to the left 
                car_left |= vehicle.isNotSafe(car_i_future_s);
            } else if (car_i_lane == current_lane + 1) { // car to the right
                car_right |= vehicle.isNotSafe(car_i_future_s);
            }
        }

        next_lane = current_lane;
        next_dist_adjust = 0.;
        
        if (car_ahead) {
            if (!car_left && current_lane > 0) { // prefer left lane for take over
                next_lane = current_lane - 1;
            } else if (!car_right && current_lane < 2) {
                next_lane = current_lane + 1;
            } else { 
                // car is ahead and we can't change lane
                next_dist_adjust = -MAX_DIST_DIFF;
            }
        } else {
            // we prefer get back to centre so that we can change lane to either left or right
            if (vehicle.speed() < MAX_SPEED_MPH) {
                next_dist_adjust = MAX_DIST_DIFF;
            }
        }
    }

    void generateTrajectory(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
        const HighwayMap& highway_map,
        const Vehicle& vehicle,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        double ref_dist = vehicle.getPrevLastDist();

        int next_lane;
        double next_dist_adjust;
        planLaneAndSpeed(
            next_lane,
            next_dist_adjust,
            vehicle,
            sensor_fusion
        );

        // step 1: get 2 last previous planned points to ensure Spline generate smooth trajectory
        double ref_x;
        double ref_y;
        double ref_yaw;
        vector<double> ptsx;
        vector<double> ptsy;

        vehicle.getPrevReferencePoints(ref_x, ref_y, ref_yaw, ptsx, ptsy);
        
        // step 2: setup target points using 30m apart in Frenet (start from current position) 
        // then convert to Cartesian
        double car_s = vehicle.s();
        std::vector<double> s_vals = {car_s + 30, car_s + 60, car_s + 90};
        std::vector<double> d_vals = std::vector<double>(3, 2 + 4 * next_lane);

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

        // step 3: plan next point in local coordinate
        
        // using Spline to plan next points in local coordinate
        splinePlanNextPoint(
            next_x_vals,
            next_y_vals,
            s,
            ref_dist,
            next_dist_adjust,
            NB_POINTS - vehicle.prev_size()
        );

        // convert back to global Catersian
        localToGlobal(next_x_vals, next_y_vals, ref_x, ref_y, ref_yaw);

        // append prev points
        vehicle.appendPrevPoints(next_x_vals, next_y_vals);
    }
}