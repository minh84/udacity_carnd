#ifndef VEHICLE_H
#define VEHICLE_H

class Vehicle {
  double _x;
  double _y; 
  double _s; 
  double _d; 
  double _yaw; 
  double _speed;
public:
  Vehicle(double x, 
          double y, 
          double s, 
          double d, 
          double yaw, 
          double speed);
};

#endif
