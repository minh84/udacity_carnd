#include "utils.h"

#include <math.h>
#include <iomanip>

using namespace std;
namespace path_planning {
  HighwayMap::HighwayMap(
      const std::vector<double>& x,
      const std::vector<double>& y,
      const std::vector<double>& s,
      const std::vector<double>& dx,
      const std::vector<double>& dy) 
      : maps_x(x)
      , maps_y(y)
      , maps_s(s)
      , maps_dx(dx)
      , maps_dy(dy)
  {
  }

   // cap distance to be in range [MAX_DIST_DIFF, MAX_DISTANCE_PER_STEP]
  double capDist(double dist) {
    return max(MAX_DIST_DIFF, min(MAX_DISTANCE_PER_STEP, dist));
  }

  // For converting back and forth between radians and degrees.
  constexpr double pi() { return M_PI; }
  double deg2rad(double x) { return x * pi() / 180; }
  double rad2deg(double x) { return x * 180 / pi(); }

  // Checks if the SocketIO event has JSON data.
  // If there is data the JSON object in string format will be returned,
  // else the empty string "" will be returned.
  std::string hasData(const std::string& s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_first_of("}");
    if (found_null != string::npos) {
      return "";
    } else if (b1 != string::npos && b2 != string::npos) {
      return s.substr(b1, b2 - b1 + 2);
    }
    return "";
  }

  double distance(double x1, double y1, double x2, double y2)
  {
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
  }

  // d is distance from center and each lane = 4m
  int getLane(double d) {
    if (d < 0 || d > 12) {
      return -1;
    }

    if (d < 4) {
      return 0;
    } else if (d < 8) {
      return 1;
    } else {
      return 2;
    }
  }

  bool isInCenterOfLane(double d, int lane) {
    double lane_center_d = 2.0 + 4.0 * lane;
    return abs(d - lane_center_d) < CENTER_THRESHOLD;
  }

  bool isChangingToLane(double d, int lane) {
    double lane_center_d = 2.0 + 4.0 * lane;
    double d_dist = abs(d - lane_center_d);
    return d_dist > 2 && d_dist < 3;
  }

  double getCarSpeed(const std::vector<double>& car_sensor) {
    double vx = car_sensor[3];
    double vy = car_sensor[4];
    return sqrt(vx*vx + vy*vy);
  }

  int ClosestWaypoint(double x, 
                      double y, 
                      const vector<double> &maps_x, 
                      const vector<double> &maps_y)
  {

    double closestLen = 100000; //large number
    int closestWaypoint = 0;

    for(int i = 0; i < maps_x.size(); i++)
    {
      double map_x = maps_x[i];
      double map_y = maps_y[i];
      double dist = distance(x,y,map_x,map_y);
      if(dist < closestLen)
      {
        closestLen = dist;
        closestWaypoint = i;
      }

    }

    return closestWaypoint;
  }

  int NextWaypoint(
    double x, 
    double y, 
    double theta, 
    const vector<double> &maps_x, 
    const vector<double> &maps_y)
  {

    int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

    double map_x = maps_x[closestWaypoint];
    double map_y = maps_y[closestWaypoint];

    double heading = atan2((map_y-y),(map_x-x));

    double angle = fabs(theta-heading);
    angle = min(2*pi() - angle, angle);

    if(angle > pi()/4)
    {
      closestWaypoint++;
      if (closestWaypoint == maps_x.size())
      {
        closestWaypoint = 0;
      }
    }

    return closestWaypoint;
  }


  // Transform from Cartesian x,y coordinates to Frenet s,d coordinates
  vector<double> getFrenet(
    double x, 
    double y, 
    double theta, 
    const vector<double> &maps_x, 
    const vector<double> &maps_y)
  {
    int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

    int prev_wp;
    prev_wp = next_wp-1;
    if(next_wp == 0)
    {
      prev_wp  = maps_x.size()-1;
    }

    double n_x = maps_x[next_wp]-maps_x[prev_wp];
    double n_y = maps_y[next_wp]-maps_y[prev_wp];
    double x_x = x - maps_x[prev_wp];
    double x_y = y - maps_y[prev_wp];

    // find the projection of x onto n
    double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
    double proj_x = proj_norm*n_x;
    double proj_y = proj_norm*n_y;

    double frenet_d = distance(x_x,x_y,proj_x,proj_y);

    //see if d value is positive or negative by comparing it to a center point

    double center_x = 1000-maps_x[prev_wp];
    double center_y = 2000-maps_y[prev_wp];
    double centerToPos = distance(center_x,center_y,x_x,x_y);
    double centerToRef = distance(center_x,center_y,proj_x,proj_y);

    if(centerToPos <= centerToRef)
    {
      frenet_d *= -1;
    }

    // calculate s value
    double frenet_s = 0;
    for(int i = 0; i < prev_wp; i++)
    {
      frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
    }

    frenet_s += distance(0,0,proj_x,proj_y);

    return {frenet_s,frenet_d};

  }

  // Transform from Frenet s,d coordinates to Cartesian x,y
  void getXY(
    double&x,
    double&y,
    double s, 
    double d, 
    const vector<double> &maps_s, 
    const vector<double> &maps_x, 
    const vector<double> &maps_y)
  {
    int prev_wp = -1;

    while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
    {
      prev_wp++;
    }

    int wp2 = (prev_wp+1)%maps_x.size();

    double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
    // the x,y,s along the segment
    double seg_s = (s-maps_s[prev_wp]);

    double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
    double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

    double perp_heading = heading-pi()/2;

    x = seg_x + d*cos(perp_heading);
    y = seg_y + d*sin(perp_heading);
  }

  void getXY(
          std::vector<double>& x_vals,
          std::vector<double>& y_vals,
    const std::vector<double>& s_vals,
    const std::vector<double>& d_vals,
    const HighwayMap& highway_map) 
  {
    size_t nb_input = s_vals.size();
    x_vals.resize(nb_input);
    y_vals.resize(nb_input);

    for(size_t i = 0; i < nb_input; ++i) {
      getXY(x_vals[i],
            y_vals[i],
            s_vals[i],
            d_vals[i],
            highway_map.maps_s,
            highway_map.maps_x,
            highway_map.maps_y);
    }
  }

  void globalToLocal(
      std::vector<double>& x_vals,
      std::vector<double>& y_vals,
      double ref_x,
      double ref_y,
      double ref_yaw
  ) {
    double cos_turn = cos(ref_yaw);
    double sin_turn = sin(ref_yaw);

    for(size_t i = 0; i < x_vals.size(); ++i) {
      // shift
      double shift_x = x_vals[i] - ref_x;
      double shift_y = y_vals[i] - ref_y;

      // turn
      x_vals[i] = shift_x * cos_turn + shift_y * sin_turn;
      y_vals[i] = -shift_x * sin_turn + shift_y * cos_turn;
    }
  }

  void localToGlobal(
      std::vector<double>& x_vals,
      std::vector<double>& y_vals,
      double ref_x,
      double ref_y,
      double ref_yaw
  ) {
    double cos_turn = cos(ref_yaw);
    double sin_turn = sin(ref_yaw);

    for(size_t i = 0; i < x_vals.size(); ++i) {
      // turn
      double shift_x = x_vals[i] * cos_turn - y_vals[i] * sin_turn;
      double shift_y = x_vals[i] * sin_turn + y_vals[i] * cos_turn;

      // shift
      x_vals[i] = ref_x + shift_x;
      y_vals[i] = ref_y + shift_y;
    }
  }

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
}