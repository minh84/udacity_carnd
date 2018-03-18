#include "vehicle.h"
#include "utils.h"

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

void Vehicle::getTrajectoryKeepLane(
  size_t nb_points,
  double s_inc,
  vector<double>& next_x_vals,
  vector<double>& next_y_vals,
  const HighwayMap& highway)
{
  // clear & reserve
  next_x_vals.resize(nb_points);
  next_y_vals.resize(nb_points);
  double next_s = _s;
  for (int i = 0; i < nb_points; ++i) {

    // increment s and keep using same current lane _d
    next_s += s_inc;
    vector<double> xy = getXY(next_s, _d, highway.maps_s, highway.maps_x, highway.maps_y);

    // append to trajectory
    next_x_vals[i] = xy[0];
    next_y_vals[i] = xy[1];
  }
}