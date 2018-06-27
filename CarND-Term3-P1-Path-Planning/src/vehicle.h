#ifndef VEHICLE_H
#define VEHICLE_H

#include "utils.h"

namespace path_planning {
  class Vehicle {
    double _x;
    double _y; 
    double _s; 
    double _d; 
    double _yaw; 
    double _speed;

    int _lane;

    // previous path-planning
    const std::vector<double>& _prev_path_x;
    const std::vector<double>& _prev_path_y;
    double _end_path_s;
    double _end_path_d;

    // forbid copy & assignment
    Vehicle(const Vehicle&) = delete;
    Vehicle& operator=(const Vehicle&) = delete;

  public:
    Vehicle(
      double x, 
      double y, 
      double s, 
      double d, 
      double yaw, 
      double speed,
      const std::vector<double>& prev_path_x,
      const std::vector<double>& prev_path_y,
      double end_path_s,
      double end_path_d);

    // getter of Vehicle: x,y,s,d, yaw and speed
    double x() const;
    double y() const;
    double s() const;
    double d() const;
    double yaw() const;
    double speed() const;

    // compute the vehicle's current lane
    int lane() const;

    // getter of previous planned points
    size_t prev_size() const;

    // append previous planned points to the begin of next_x_vals, next_y_vals (inplace)
    void appendPrevPoints(
      std::vector<double>& next_x_vals,
      std::vector<double>& next_y_vals
    ) const;

    // get s when the car passed all previous planned points
    // in the case there is no previous planned points we return current s
    double end_path_s() const;

    // compute the previous last distance of planned points
    // in the first step, there is no planned points, we return 0
    double getPrevLastDist() const;

    // compute the 2 last points which is used in generating trajectory with Spline
    // by using the 2 last points, we ensure the smoothness of generated trajectory
    void getPrevReferencePoints(
      double& ref_x,
      double& ref_y,
      double& ref_yaw,
      std::vector<double>& ptsx,
      std::vector<double>& ptsy
    ) const;

    // compute Spline points 
    void getSplinePoints(
      std::vector<double>& ptsx,
      std::vector<double>& ptsy,
      const HighwayMap& highway_map,
      int lane,
      double s_step
    ) const;

    // given a car-sensor, we estimate the future position in Frenet after the car
    // passed all previous planned points
    double getFuturePosition(
      const std::vector<double>& car_sensor
    ) const;

    // check in Frenet is a given s is ahead of end_path_s() and distance to end_path_s() < SAFETY_DIST
    bool isAhead(double s) const;

    // check if it's within non-safe to change lane
    bool isNotSafeToChangeLane(double s) const;

    // check if change lane is not better
    bool isNotBetterToChangeLane(
      const std::vector<double>& car_in_same_lane,
      const std::vector<double>& car_in_other_lane) const;

    // check in Frenet if a given s is within SAFETY_DIST with the end_path_s()
    bool isNotSafe(double s) const;
  };
}


#endif
