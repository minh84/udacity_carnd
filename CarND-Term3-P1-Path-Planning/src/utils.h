#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

namespace utils {
  constexpr double pi();
  double deg2rad(double x);
  double rad2deg(double x);

  std::string hasData(const std::string& s);

  double distance(double x1, double y1, double x2, double y2);

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
  std::vector<double> getXY(
    double s, 
    double d, 
    const std::vector<double> &maps_s, 
    const std::vector<double> &maps_x, 
    const std::vector<double> &maps_y);

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
}

#endif
