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

  double x() const;
  double y() const;
  double s() const;
  double d() const;
  double yaw() const;
  double speed() const;

};

#endif
