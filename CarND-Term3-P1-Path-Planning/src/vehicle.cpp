#include "vehicle.h"
#include "utils.h"
#include <math.h>

using namespace utils;
using namespace std;
Vehicle::Vehicle(
  double x, 
  double y, 
  double s, 
  double d, 
  double yaw, 
  double speed) 
  : _x(x)
  , _y(y)
  , _s(s)
  , _d(d)
  , _yaw(yaw)
  , _speed(speed)
{
}

double Vehicle::x() const {
  return _x;
}
double Vehicle::y() const {
  return _y;
}
double Vehicle::s() const {
  return _s;
}
double Vehicle::d() const {
  return _d;
}
double Vehicle::yaw() const {
  return _yaw;
}
double Vehicle::speed() const {
  return _speed;
}