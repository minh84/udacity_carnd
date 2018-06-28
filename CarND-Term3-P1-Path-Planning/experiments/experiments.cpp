#include "experiments.h"
#include "spline.h"
#include <math.h>
#include <iostream>

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

    int getLaneOffset(
        int i_ahead,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion,
        double look_dist 
    ) {
        if (i_ahead == -1) return 0;

        int lane_offset = 0;

        // get lane, current & future position
        int lane = car.lane();
        double s_current = car.s();
        double s_future = car.end_path_s();

        bool can_change_left = (lane > 0);
        bool can_change_right = (lane < 2);

        double nearest_left_ahead = 10000;
        double nearest_right_ahead = 10000;
        auto& car_ahead = sensor_fusion[i_ahead];

        for (int i = 0; i < sensor_fusion.size(); ++i) {
            if (!can_change_left && !can_change_right) {
                break;
            }

            auto& car_i = sensor_fusion[i]; // id, x, y, vx, vy, s, d
            int car_i_lane = getLane(car_i[6]);
            if (car_i_lane == -1 
                || car_i_lane == lane
                || abs(car_i_lane - lane) == 2 ) {
                continue;
            }

            double dist = abs(car_i[5] - s_current);

            if (car_i[5] > s_current) { // get nearest ahead
                if (car_i_lane == lane - 1) {
                    nearest_left_ahead = min(dist, nearest_left_ahead);
                } else {
                    nearest_right_ahead = min(dist, nearest_right_ahead);
                }
            }
            
            if (dist > look_dist) { // out of look range
                continue;
            }

            double future_dist = abs(car.getFuturePosition(car_i) - s_future);
            bool isNotSafe = car.isNotSafeToChangeLane(car_i[5]);
            bool isNotBetter = car.isNotBetterToChangeLane(car_ahead, car_i);
            if (isNotSafe || isNotBetter) {
                // a car in safety range => can't change lane
                if (car_i_lane == lane - 1) {
                    can_change_left = false;
                } else {
                    can_change_right = false;
                }
                cout << "car " << i << " isNotSafe=" << isNotSafe 
                     << ", isNotBetter=" << isNotBetter << "\n";
            } 
        }

        // prefer turn to the lane where car ahead in that lane is further
        if (can_change_left && can_change_right) {
            return (nearest_left_ahead < nearest_right_ahead)
                   ? 1
                   : -1; 
        } else {
            if (can_change_left) {
                return -1;
            }
            if (can_change_right) {
                return 1;
            }

            return 0;
        }
    }
}

namespace path_planning {

    void logToFile(
        std::ofstream& outfile,
        int step,
        const std::string& tag,
        double value)
    {
        outfile << step << ";" << tag << ";" 
                << std::fixed << std::setprecision(3) << value << "\n"; 
    }

    void logToFile(
        std::ofstream& outfile,
        int step,
        const std::string& tag,
        const std::vector<double>& values)
    {
        outfile << step << ";" << tag << ";[";
        outfile << std::fixed << std::setprecision(3);
        if (!values.empty()) {
            outfile << values[0];
            for (int i = 1; i < values.size(); ++i) {
                outfile << "," << values[i];
            }
        }
        outfile << "]\n";
    }
    
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
        int lane = car.lane();
        
        // step 1: get 2 last previous planned points to ensure Spline generate smooth trajectory
        double ref_x;
        double ref_y;
        double ref_yaw;
        vector<double> ptsx;
        vector<double> ptsy;

        getRefPointsForSpline(ref_x, ref_y, ref_yaw, ptsx, ptsy, highway_map, car, lane);

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

    int getCarAhead(
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        int i_ahead = -1;
        double min_s_ahead = 10000;
        for(size_t i = 0; i < sensor_fusion.size(); ++i) {
            auto& car_i = sensor_fusion[i]; // id, x, y, vx, vy, s, d

            int car_i_lane = getLane(car_i[6]);

            if (car_i_lane == car.lane()) {
                double future_s = car.getFuturePosition(car_i);

                if (car.isAhead(future_s) && future_s < min_s_ahead) {
                    min_s_ahead = future_s;
                    i_ahead = i;
                }
            }
        }

        return i_ahead;
    }

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

    void getTrajectoryKeepLaneNoRed(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              double& ref_v,
        const HighwayMap& highway_map,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion,
        size_t nb_points,
        double max_acc 
    ) {
        int target_lane = car.lane();

        // get nearest car ahead (-1 means no car ahead)
        int i_ahead = getCarAhead(car, sensor_fusion);

        // compute target speed: if no car ahead => max speed, otherwise speed of the car ahead
        double target_speed = (i_ahead == -1)
                             ? MAX_SPEED_MPS
                             : getCarSpeed(sensor_fusion[i_ahead]);

        getTrajectoryGivenLaneAndSpeed(
            next_x_vals,
            next_y_vals,
            ref_v,
            highway_map,
            car,
            nb_points,
            max_acc,
            target_lane,
            target_speed
        );
    }

    void getTrajectoryAllowChangeLane(
              std::vector<double>& next_x_vals,
              std::vector<double>& next_y_vals,
              double& ref_v,
        const HighwayMap& highway_map,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion,
        size_t nb_points,
        double max_acc,
        double look_dist
    ) {
        int target_lane = car.lane();
        double target_speed = ref_v;
        int lane_offset = 0;
        // get nearest car ahead (-1 means no car ahead)
        int i_ahead = getCarAhead(car, sensor_fusion);

        if (i_ahead == -1) {
            target_speed = MAX_SPEED_MPS;
        } else {
            lane_offset = getLaneOffset(i_ahead, car, sensor_fusion, look_dist);
            if (lane_offset == 0) {
                // we can't turn left or right we have to slow down and following car ahead
                target_speed = getCarSpeed(sensor_fusion[i_ahead]);
            } 
            target_lane += lane_offset;
        }

        cout << "i_ahead = " << i_ahead << ", lane_offset = " << lane_offset << "\n";

        // get the trajectory
        getTrajectoryGivenLaneAndSpeed(
            next_x_vals,
            next_y_vals,
            ref_v,
            highway_map,
            car,
            nb_points,
            max_acc,
            target_lane,
            target_speed
        );
    }
}