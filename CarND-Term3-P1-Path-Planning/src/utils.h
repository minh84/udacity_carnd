#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

namespace path_planning {
  // Define const

  // number of point
  static const size_t NB_POINTS = 50;

  // max speed is 50 MPH, then we convert to MPS
  static const double MAX_SPEED_MPH = 49.5;
  static const double MPH_TO_MPS = 0.44704;
  static const double MAX_SPEED_MPS = MAX_SPEED_MPH * MPH_TO_MPS;

  // each time step is 20ms = 0.02 second
  static const double TIME_STEP = 0.02;
  static const double MAX_DISTANCE_PER_STEP = TIME_STEP * MAX_SPEED_MPS;

  // max acceleration is 10m/s^2
  static const double MAX_ACC = 5; // m/s2
  static const double MAX_SPEED_DIFF = MAX_ACC * TIME_STEP;
  static const double MAX_DIST_DIFF = MAX_SPEED_DIFF * TIME_STEP;

  // safety distance 30m
  static const double SAFETY_DIST = 30; 

  // Store highways map to make it easier to pass as an argument
  // we don't 
  struct HighwayMap {
    std::vector<double> maps_x;
    std::vector<double> maps_y;
    std::vector<double> maps_s;
    std::vector<double> maps_dx;
    std::vector<double> maps_dy;
    HighwayMap(
      const std::vector<double>& x,
      const std::vector<double>& y,
      const std::vector<double>& s,
      const std::vector<double>& dx,
      const std::vector<double>& dy);
  };

   // cap distance to be in range [MAX_DIST_DIFF, MAX_DISTANCE_PER_STEP]
  double capDist(double dist);

  constexpr double pi();
  double deg2rad(double x);
  double rad2deg(double x);

  std::string hasData(const std::string& s);

  double distance(double x1, double y1, double x2, double y2);

  int getLane(double d);
  double getCarSpeed(const std::vector<double>& car_sensor);

  int ClosestWaypoint(
      double x, 
      double y, 
      const std::vector<double> &maps_x, 
      const std::vector<double> &maps_y);

  int NextWaypoint(
    double x, 
    double y, 
    double theta, 
    const std::vector<double> &maps_x, 
    const std::vector<double> &maps_y);

  // Transform from Cartesian x,y coordinates to Frenet s,d coordinates
  std::vector<double> getFrenet(
    double x, 
    double y, 
    double theta, 
    const std::vector<double> &maps_x, 
    const std::vector<double> &maps_y);

  // Transform from Frenet s,d coordinates to Cartesian x,y
  void getXY(
    double&x,
    double&y,
    double s, 
    double d, 
    const std::vector<double> &maps_s, 
    const std::vector<double> &maps_x, 
    const std::vector<double> &maps_y);

  void getXY(
          std::vector<double>& x_vals,
          std::vector<double>& y_vals,
    const std::vector<double>& s_vals,
    const std::vector<double>& d_vals,
    const HighwayMap& highway_map);

  // Convert from global => local coordinate (inplace)
  void globalToLocal(
      std::vector<double>& x_vals,
      std::vector<double>& y_vals,
      double ref_x,
      double ref_y,
      double ref_yaw
  );
  
  void localToGlobal(
      std::vector<double>& x_vals,
      std::vector<double>& y_vals,
      double ref_x,
      double ref_y,
      double ref_yaw
  );

  void logToFile(
    std::ofstream& outfile,
    int step,
    const std::string& tag,
    double value);

  void logToFile(
    std::ofstream& outfile,
    int step,
    const std::string& tag,
    const std::vector<double>& values);
}

#endif
