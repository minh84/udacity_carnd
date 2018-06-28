#include "trajectory.h"
#include "spline.h"

using namespace std;
using namespace path_planning;

namespace {
    void getRefPointsForSpline(
              double& ref_x,
              double& ref_y,
              double& ref_yaw,
              vector<double>& ptsx,
              vector<double>& ptsy,
        const HighwayMap& highway_map,
        const Vehicle& car,
        int target_lane
    ) {
        car.getPrevReferencePoints(ref_x, ref_y, ref_yaw, ptsx, ptsy);
        
        // step 2: setup target points using 30m apart in Frenet (start from current position) 
        // then convert to Cartesian
        std::vector<double> x_vals;
        std::vector<double> y_vals;
        car.getSplinePoints(x_vals, y_vals, highway_map, target_lane, 30);

        // points inlcudes previous point and future points to ensure the smoothness
        ptsx.insert(ptsx.end(), x_vals.begin(), x_vals.end());
        ptsy.insert(ptsy.end(), y_vals.begin(), y_vals.end());   
    }

    void planPointFollowingSpeed(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              double& ref_v,
        const tk::spline& s,
        double target_speed,
        size_t nb_points,
        double max_acc
    ) {
        next_x_vals.resize(nb_points);
        next_y_vals.resize(nb_points);

         // calculate point in 30m ahead (using car's local coordinate)
        double local_target_x = 30;
        double local_target_y = s(local_target_x);
        double target_dist = sqrt(local_target_x * local_target_x + local_target_y * local_target_y);

        double diff = abs(ref_v - target_speed);
        size_t i = 0;
        double x = 0.;
        double max_speed_diff = max_acc * TIME_STEP;

        // while speed not close enough to target-speed => we adjust the speed
        while ((diff > max_speed_diff) && (i < nb_points)) {
            if (ref_v < target_speed) {
                ref_v += max_speed_diff;
            } else {
                ref_v -= max_speed_diff;
            }

            diff -= max_speed_diff;
            double fraction = ref_v * TIME_STEP / target_dist;
            x += fraction * local_target_x;

            next_x_vals[i] = x;
            next_y_vals[i] = s(x);
            ++i;
        }
        
        // when it's close enough we set it to target-speed
        if (i < nb_points) {
            ref_v = target_speed;
            double x_step = local_target_x * ref_v * TIME_STEP / target_dist;

            for (;i < nb_points;++i) {
                x += x_step;
                next_x_vals[i] = x;
                next_y_vals[i] = s(x);
            }
        }
        
    }
}

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
    ) {
        // step 1: get Spline
        double ref_x;
        double ref_y;
        double ref_yaw;
        vector<double> ptsx;
        vector<double> ptsy;

        getRefPointsForSpline(ref_x, ref_y, ref_yaw, ptsx, ptsy, highway_map, car, target_lane);

        // transform to local coordinate
        globalToLocal(ptsx, ptsy, ref_x, ref_y, ref_yaw);

        // spline
        tk::spline s;
        s.set_points(ptsx, ptsy);

        // plan next point
        size_t nb_next_point = nb_points - car.prev_size();

        // plan to get to target speed
        planPointFollowingSpeed(next_x_vals, next_y_vals, ref_v, s, target_speed, nb_next_point, max_acc);
        localToGlobal(next_x_vals, next_y_vals, ref_x, ref_y, ref_yaw);

        // append previous points
        car.appendPrevPoints(next_x_vals, next_y_vals);
    }
}