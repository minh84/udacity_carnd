#include "behavior.h"
#include "spline.h"
#include <math.h>

using namespace path_planning;
using namespace std;

namespace {
    struct CarNearBy {
        double _dist_s;
        int _i_nearest;
        

        CarNearBy(
        )   : _dist_s(10000)
            , _i_nearest(-1)
        {
        }

        void update(
            int i, 
            double dist) 
        {
            if (dist < _dist_s) {
                _dist_s = dist;
                _i_nearest = i;
            }
        }

        bool isBlock(double safety_dist) const {
            return (_i_nearest != -1) && (_dist_s < safety_dist);
        }

        double futureDist(
            const Vehicle& car,
            const std::vector<std::vector<double>>& sensor_fusion) const {
            return (_i_nearest==-1) 
                    ? 20000
                    : abs(car.getFuturePosition(sensor_fusion[_i_nearest]) - car.end_path_s());
        }

        bool isImproveDist(
            const Vehicle& car,
            const std::vector<std::vector<double>>& sensor_fusion,
            double dist_improve) const {
            return futureDist(car, sensor_fusion) > _dist_s + dist_improve;
        }

        double getSpeedNearestCar(const std::vector<std::vector<double>>& sensor_fusion) {
            return (_i_nearest==-1) 
                    ? MAX_SPEED_MPS
                    : getCarSpeed(sensor_fusion[_i_nearest]);
        }

        bool nearestCarIsChangingToLane(
            int lane,
            const std::vector<std::vector<double>>& sensor_fusion) const {
            return (_i_nearest == -1)
                    ? true
                    : isChangingToLane(sensor_fusion[_i_nearest][6], lane);

        }
    };

    bool canChangeToLane(
        const CarNearBy& carAhead,
        const CarNearBy& carBehind,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        if (carAhead._lane == -1) {
            return false;
        }

        if (carAhead.isBlock(SAFETY_DIST_CHANGE_LANE_AHEAD) 
            || carBehind.isBlock(SAFETY_DIST_CHANGE_LANE_BEHIND)) {
            return false;
        }

        if (!carAhead.isImproveDist(car, sensor_fusion, DIST_IMPROVE)
            || !carBehind.isImproveDist(car, sensor_fusion, DIST_IMPROVE)) {
            return false;
        }

        return true;
    }

    double getAheadSpeed(
        int target_lane,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        CarNearBy car_ahead(car.s(), target_lane, true);

        for(int i = 0; i < sensor_fusion.size(); ++i) {
            double s_i = sensor_fusion[i][5];
            int lane_i = getLane(sensor_fusion[i][6]);
            car_ahead.update(i, lane_i, s_i);
        }

        if (car_ahead.isBlock(SAFETY_DIST)) {
            return car_ahead.getSpeedNearestCar(sensor_fusion);
        } else {
            return MAX_SPEED_MPS;
        }
    }

    int getLaneOffset(
        int current_lane,
        const CarNearBy& l_ahead,
        const CarNearBy& r_ahead,
        const CarNearBy& l_behind,
        const CarNearBy& r_behind,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        bool can_change_left = (current_lane > 0) && canChangeToLane(l_ahead, l_behind, car, sensor_fusion);
        bool can_change_right = (current_lane < 2) && canChangeToLane(r_ahead, r_behind, car, sensor_fusion);

        if (can_change_left && can_change_right) {
            if (r_ahead.futureDist(car, sensor_fusion) > l_ahead.futureDist(car, sensor_fusion)) {
                return 1;
            } else {
                return -1;
            }
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

    bool canChangeToLane(
        const CarNearBy& carAhead,
        const CarNearBy& carAheadInLane,
        const CarNearBy& carBehindInLane,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        if (carAhead.isBlock(SAFETY_DIST_CHANGE_LANE_AHEAD) 
            || carBehind.isBlock(SAFETY_DIST_CHANGE_LANE_BEHIND)) {
            return false;
        }

        if (!carAhead.isImproveDist(car, sensor_fusion, DIST_IMPROVE)
            || !carBehind.isImproveDist(car, sensor_fusion, DIST_IMPROVE)) {
            return false;
        }

        if (carAheadInLane.getSpeedNearestCar(sensor_fusion) < carAhead.getSpeedNearestCar(sensor_fusion)) {
            return false;
        } 

        return true;
    }

    bool canChangeToLane(
        int target_lane,
        const vector<CarNearBy>& cars_ahead,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        return false;
    }

    void getLaneAndSpeed(
        int& target_lane,
        double& target_speed,
        const vector<CarNearBy>& cars_ahead,
        const vector<CarNearBy>& cars_behind,
        const Vehicle& car,
        const std::vector<std::vector<double>>& sensor_fusion
    ) {
        // check if we get blocked
        //      => try to change lane
        //      => otherwise keep lane and reduce speed
        // we assume that car in other lane might change to our lane in no car ahead block us
        int current_lane = car.lane();
        target_lane = current_lane;
        bool is_blocked_ahead = cars_ahead[current_lane].isBlock(SAFETY_DIST);

        bool can_change_left = (current_lane > 0 
                                && canChangeToLane(current_lane - 1, cars_ahead, car, sensor_fusion));
        bool can_change_right = (current_lane < 2 
                                && canChangeToLane(current_lane + 1, cars_ahead, car, sensor_fusion));
        bool is_blocked_left  = false;
        bool is_blocked_right = false;
        if (!is_blocked_ahead) {
            is_blocked_left |= (current_lane > 0 
                                && cars_ahead[current_lane - 1].isBlock(SAFETY_DIST)
                                && cars_ahead[current_lane - 1].nearestCarIsChangingToLane(current_lane));
            
            is_blocked_right |= (current_lane < 2
                                && cars_ahead[current_lane + 1].isBlock(SAFETY_DIST)
                                && cars_ahead[current_lane + 1].nearestCarIsChangingToLane(current_lane));

            if (is_blocked_left) {
                // try to change to right lane
                if (current_lane < 2 
                    && canChangeToLane(current_lane + 1, cars_ahead, car, sensor_fusion)) {
                    target_lane = current_lane + 1;
                    target_speed = cars_ahead[target_lane].getSpeedNearestCar(sensor_fusion);
                } else {
                    target_speed = cars_ahead[current_lane - 1].getSpeedNearestCar(sensor_fusion);
                }
            } else if (is_blocked_right) {
                // try to change to left lane
                if  {
                    target_lane = current_lane - 1;
                    target_speed = cars_ahead[target_lane].getSpeedNearestCar(sensor_fusion);
                } else {
                    target_speed = cars_ahead[current_lane + 1].getSpeedNearestCar(sensor_fusion);
                }
            } else {
                // no block keep the max speed
                target_speed = MAX_SPEED_MPS;
            }
        } else { // already block

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

        double target_speed = _ref_v;
        
        // check if any car is changing lane mode: we keep _taget_lane and compute the _target_speed
        if (!isInCenterOfLane(car.d(), _target_lane)) {
            target_speed = getAheadSpeed(_target_lane, car, sensor_fusion);
        } else {
            // otherwise check surrounding cars
            vector<CarNearBy> carsAhead(3);
            vector<CarNearBy> carsBehind(3);
            double s_behind = car.s() - _look_behind_dist;
            double s_ahead = car.s() + _look_ahead_dist;

            for(int i = 0; i < sensor_fusion.size(); ++i) {
                double s_i = sensor_fusion[i][5];
                int lane_i = getLane(sensor_fusion[i][6]);
                if (lane_i == -1
                    || s_i < s_behind
                    || s_i > s_ahead) {
                    continue;
                }

                if (s_i < car.s()) {
                    carsBehind[lane_i].update(i, car.s() - s_i);
                } else {
                    carsAhead[lane_i].update(i, s_i - car.s());
                }
            }
        }

        getTrajectoryGivenLaneAndSpeed(
            next_x_vals,
            next_y_vals,
            _ref_v,
            _highway,
            car,
            NB_POINTS,
            MAX_ACC,
            _target_lane,
            target_speed
        );
    }
}