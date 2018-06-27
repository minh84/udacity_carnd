#include "behavior.h"
#include "spline.h"
#include <math.h>
#include <iostream>

using namespace path_planning;
using namespace std;

namespace {
    struct CarNearBy {
        double _dist_s;
        int _i_nearest;
        
        double _speed;
        double _future_dist_s;
        int    _lane_offset;

        CarNearBy(
        )   : _dist_s(10000)
            , _i_nearest(-1)
            , _speed(MAX_SPEED_MPS)
            , _future_dist_s(20000)
            , _lane_offset(0)
        {
        }

        void update(int i, double dist) 
        {
            if (dist < _dist_s) {
                _dist_s = dist;
                _i_nearest = i;
            }
        }

        // this should be called only once
        void finalUpdate(
            int current_lane,
            const Vehicle& car, 
            const std::vector<std::vector<double>>& sensor_fusion) {
            
            if (_i_nearest != -1) {
                auto& nearest_car = sensor_fusion[_i_nearest]; // id, x, y, vx, vy, s, d
                _speed = getCarSpeed(nearest_car);
                _future_dist_s = abs(car.getFuturePosition(nearest_car) - car.end_path_s());

                // check if our nearest car is changing to other lane
                double lane_center_d = 2.0 + 4.0 * current_lane;
                double d = nearest_car[6];
                if (current_lane > 0 && d < lane_center_d - CHANGE_LANE_THRESHOLD) {
                    _lane_offset = -1;
                } else if (current_lane < 2 && d > lane_center_d + CHANGE_LANE_THRESHOLD) {
                    _lane_offset = 1;
                }
            }
        }

        // get speed ahead
        double limitSpeed(double current_speed) const {
            return min(_speed, current_speed);
        }

        int laneOffset() const {
            return _lane_offset;
        }

        bool isNotImproveDist() const {
            return _future_dist_s < _dist_s + IMPROVE_DIST_THRESHOLD;
        }

        bool isWithin(double dist) const {
            return _dist_s < dist;
        }
    };

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

    // limit speed by car ahead
    void limitSpeed(
              vector<double>& speed,
        const vector<CarNearBy>& carsAhead
    ) {
        for(int i = 0; i < NB_LANES; ++i) {
            auto& carAhead = carsAhead[i];
            speed[i] = carAhead.limitSpeed(speed[i]);

            // if car ahead is changing into other lane then it will 
            // cause a speed limit in the other lane too
            if (carAhead.laneOffset()) {
                int i_next = i + carAhead.laneOffset(); 
                speed[i_next] = carAhead.limitSpeed(speed[i_next]);
            }
        }
    }

    // check if we should change to lane
    bool canChangeTo(
        int target_lane,
        const vector<CarNearBy>& carsAhead,
        const vector<CarNearBy>& carsBehind
    ) {
        auto& carAhead = carsAhead[target_lane];
        auto& carBehind = carsBehind[target_lane];

        if (carAhead.isWithin(SAFETY_DIST_CHANGE_LANE_AHEAD)
            || carBehind.isWithin(SAFETY_DIST_CHANGE_LANE_BEHIND)) 
        {
            return false;
        }

        if (carAhead.isNotImproveDist() || carBehind.isNotImproveDist()) 
        {
            return false;
        }

        return true;
    }

    int computeLaneOffset(
        const Vehicle& car,
        const int current_lane,
        const vector<double>& speed,
        const vector<CarNearBy>& carsAhead,
        const vector<CarNearBy>& carsBehind
    ) {
        int lane_offset = 0;
        // check if we are not in changing-lane mode
        if (isInCenterOfLane(car.d(), current_lane) 
                && speed[current_lane] < MAX_SPEED_MPS) {
            bool could_change_left = (current_lane > 0) 
                                        && (speed[current_lane - 1] > speed[current_lane])
                                        && canChangeTo(current_lane - 1, carsAhead, carsBehind);
            bool could_change_right = (current_lane < 2)
                                        && (speed[current_lane + 1] > speed[current_lane])
                                        && canChangeTo(current_lane + 1, carsAhead, carsBehind);

            if (could_change_left && could_change_right) {
                lane_offset = (speed[current_lane + 1] > speed[current_lane - 1])
                                ? 1
                                : -1;
            } else {
                if (could_change_left) {
                    lane_offset = -1;
                }

                if (could_change_right) {
                    lane_offset = 1;
                }
            }

            /*cout << "current_lane =" << current_lane 
                 << ", could_change_left =" << could_change_left  
                 << ", could_change_right =" << could_change_right << "\n";*/
        }
        return lane_offset;
    }
}

namespace path_planning {

    Behavior::Behavior(
        const HighwayMap& highway,
        double look_ahead_dist,
        double look_behind_dist
    ) : _highway(highway)
      , _look_ahead_dist(look_ahead_dist)   
      , _look_behind_dist(look_behind_dist)
    {
        reset();
    }

    void Behavior::reset() {
        _ref_v = 0.;
        _target_lane = 1;
        _step = 0;
    }

    void Behavior::generateTrajectory(
            std::vector<double>& next_x_vals,
            std::vector<double>& next_y_vals,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        // keep cache of current step
        ++_step;  

        // get the list of surrounding car (where s-car.s() in the range [-look_behind_dist, lock_ahead_dist])
        vector<CarNearBy> carsAhead(NB_LANES);
        vector<CarNearBy> carsBehind(NB_LANES);
        double s_behind = car.s() - _look_behind_dist;
        double s_ahead = car.s() + _look_ahead_dist;

        for(int i = 0; i < sensor_fusion.size(); ++i) {
            double s_i = sensor_fusion[i][5];
            int lane_i = getLane(sensor_fusion[i][6]);

            // check if car in the interested range
            if (s_i < s_behind || s_i > s_ahead || lane_i == -1) {
                continue;
            }

            if (s_i < car.s()) {
                carsBehind[lane_i].update(i, car.s() - s_i);
            } else {
                carsAhead[lane_i].update(i, s_i - car.s());
            }
        }

        // update speed & future dist
        for (int i = 0; i < NB_LANES; ++i) {
            carsBehind[i].finalUpdate(i, car, sensor_fusion);
            carsAhead[i].finalUpdate(i, car, sensor_fusion);
        }

        // evaluate speed in each lane
        std::vector<double> speed(NB_LANES, MAX_SPEED_MPS);
        limitSpeed(speed, carsAhead);

        int lane_offset = computeLaneOffset(car, _target_lane, speed, carsAhead, carsBehind);  
        _target_lane += lane_offset;


        getTrajectoryGivenLaneAndSpeed(
            next_x_vals,
            next_y_vals,
            _ref_v,
            _highway,
            car,
            NB_POINTS,
            MAX_ACC,
            _target_lane,
            speed[_target_lane]
        );
    }
}