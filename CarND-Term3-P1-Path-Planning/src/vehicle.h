#ifndef VEHICLE_H
#define VEHICLE_H

#include "utils.h"

class Vehicle {
  double _x;
  double _y; 
  double _s; 
  double _d; 
  double _yaw; 
  double _speed;
public:
  Vehicle(
    double x, 
    double y, 
    double s, 
    double d, 
    double yaw, 
    double speed);

  void getTrajectoryKeepLane(
    size_t nb_points,
    double s_inc,
    std::vector<double>& next_x_vals,
    std::vector<double>& next_y_vals,
    const utils::HighwayMap& highway);
};

#endif
