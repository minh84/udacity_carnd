#include "vehicle.h"
#include "utils.h"
#include <math.h>

using namespace std;

namespace path_planning {
  Vehicle::Vehicle(
    double x, 
    double y, 
    double s, 
    double d, 
    double yaw, 
    double speed,
    const std::vector<double>& prev_path_x,
    const std::vector<double>& prev_path_y,
    double end_path_s,
    double end_path_d) 
    : _x(x)
    , _y(y)
    , _s(s)
    , _d(d)
    , _yaw(yaw)
    , _speed(speed)
    , _prev_path_x(prev_path_x)
    , _prev_path_y(prev_path_y)
    , _end_path_s(end_path_s)
    , _end_path_d(end_path_d)
  {
    _lane = getLane(_d);
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

  int Vehicle::lane() const {
    return _lane;
  }

  size_t Vehicle::prev_size() const {
    return _prev_path_x.size();
  }

  void Vehicle::appendPrevPoints(
      std::vector<double>& next_x_vals,
      std::vector<double>& next_y_vals
    ) const {
    if (!_prev_path_x.empty()) {
      next_x_vals.insert(next_x_vals.begin(), _prev_path_x.begin(), _prev_path_x.end());
      next_y_vals.insert(next_y_vals.begin(), _prev_path_y.begin(), _prev_path_y.end());
    }
  }

  double Vehicle::end_path_s() const {
    return _prev_path_x.empty()
          ? _s
          : _end_path_s;
  }

  double Vehicle::getPrevLastDist() const {
    size_t prev_size = _prev_path_x.size();
    return (_prev_path_x.size() < 2)
          ? 0.0
          : distance(_prev_path_x[prev_size - 2],
                     _prev_path_y[prev_size - 2],
                     _prev_path_x[prev_size - 1],
                     _prev_path_y[prev_size - 1]);
  }

  void Vehicle::getPrevReferencePoints(
      double& ref_x,
      double& ref_y,
      double& ref_yaw,
      std::vector<double>& ptsx,
      std::vector<double>& ptsy
    ) const {
    // we use previous planned points as reference
    size_t prev_size = _prev_path_x.size();

    double prev_x, prev_y;

    if (prev_size < 2) {
        ref_x = _x;
        ref_y = _y;
        ref_yaw = deg2rad(_yaw);
        prev_x = ref_x - cos(ref_yaw);
        prev_y = ref_y - sin(ref_yaw);
    } else {
        ref_x  = _prev_path_x[prev_size - 1];
        ref_y  = _prev_path_y[prev_size - 1];
        prev_x = _prev_path_x[prev_size - 2];
        prev_y = _prev_path_y[prev_size - 2];
        ref_yaw = atan2(ref_y - prev_y, ref_x - prev_x);
    }

    // use 2 previous points to ensure smoothness since Spline is continuous in values and derivatives
    ptsx.push_back(prev_x); 
    ptsx.push_back(ref_x);

    ptsy.push_back(prev_y); 
    ptsy.push_back(ref_y);
  }

  void Vehicle::getSplinePoints(
      std::vector<double>& ptsx,
      std::vector<double>& ptsy,
      const HighwayMap& highway_map,
      int lane,
      double s_step
    ) const {
    
    double d = 2 + 4 * lane;
    double car_s = end_path_s();

    // we expect that at car_s + s_step, we can change to lane d = 2 + 4 * lane
    std::vector<double> s_vals = {car_s + s_step, car_s + 2*s_step, car_s + 3*s_step};
    std::vector<double> d_vals = {d, d, d};

    getXY(ptsx, ptsy, s_vals, d_vals, highway_map);
  } 

  double Vehicle::getFuturePosition(
      const std::vector<double>& car_sensor
    ) const {
    // extract x,y,vx,vy,s,d from car_sensor
    double vx = car_sensor[3];
    double vy = car_sensor[4];
    double s = car_sensor[5];

    // the car's speed
    double car_speed = sqrt(vx * vx + vy * vy);
    
    // number of planned points
    size_t prev_size = _prev_path_x.size();
    return s + prev_size * TIME_STEP * car_speed;
  }
}
