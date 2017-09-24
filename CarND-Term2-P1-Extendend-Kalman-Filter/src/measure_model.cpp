#include "measure_model.h"

LaserMeasure::LaserMeasure() 
  : MeasureModel(4, 2)
{
  Hj_ << 1, 0, 0, 0,
         0, 1, 0, 0;
}

const Eigen::VectorXd& LaserMeasure::measure(const Eigen::VectorXd& x) {
  /**
   * Laser-measure returns px, py from state x = (px, py, vx, vy)
   */
  hx_[0] = x[0];
  hx_[1] = x[1];
  return hx_;
}

const Eigen::MatrixXd& LaserMeasure::jacobian(const Eigen::VectorXd& /*x*/) {
  return Hj_;
}

Eigen::VectorXd LaserMeasure::computeState0(const Eigen::VectorXd& z) const {
  Eigen::VectorXd x0 = Eigen::VectorXd(4);
  x0 << z[0], z[1], 0, 0;
  
  return x0;
}

RadarMeasure::RadarMeasure() 
: MeasureModel(4, 3)
{
}

const Eigen::VectorXd& RadarMeasure::measure(const Eigen::VectorXd& x) {
  /**
   Radar returns (rho, phi, rhodot) from state x = (px, py, vx, vy) using following equations
      rho = sqrt(px*px + py*py)
      phi = atan2(py, px) 
      rhodot = (px * vx + py * vy) / rho;
   Note that we have to handle the case rho = 0.
   */
  const double eps = 1e-6;
  double rho = sqrt(x[0] * x[0] + x[1] * x[1]);

  hx_[0] = rho;
  hx_[1] = atan2(x[1], x[0]);
  // if rho > eps, we compute rhodot, otherwise, we use the previous rhodot
  if (rho > eps) {
    hx_[2] = (x[0] * x[2] + x[1] * x[3]) / rho;
  }

  return hx_;
}

const Eigen::MatrixXd& RadarMeasure::jacobian(const Eigen::VectorXd& x) {
  const double eps = 1e-6;
  double rho2 = x[0]*x[0] + x[1]*x[1];  
  double rho = sqrt(rho2);

  // we only re-compute Hj_ if rho > eps
  if (rho > eps) {
    double rho3 = rho * rho2;
    double cross = x[2]*x[1] - x[3]*x[0];

    Hj_ << x[0]/rho,           x[1]/rho,  0, 0,
          -x[1]/rho2,          x[0]/rho2, 0, 0,
           x[1] * cross/rho3, -x[0] * cross/rho3, x[0]/rho, x[1]/rho;
  } 

  return Hj_;
}

Eigen::VectorXd RadarMeasure::computeState0(const Eigen::VectorXd& z) const {
  /**
  We have the measurement
    z = (rho, phi, rhodot)
  We have then
    px = z * cos(phi)
    py = z * sin(phi)
  We can't compute (vx, vy) from polar measurement but we can approximate
    vx = rhodot * cos(phi)
    vy = rhodot * sin(phi)
   */
  double cosphi = cos(z[1]);
  double sinphi = sin(z[1]);

  Eigen::VectorXd x0 = Eigen::VectorXd(4);
  x0 << z[0] * cosphi, z[0] * sinphi, z[2] * cosphi, z[2] * sinphi;
  
  return x0;
}

void RadarMeasure::normalize(Eigen::VectorXd& y) const {
  const double twoPi = 2.*M_PI;
  while(y[1] >= M_PI)  y[1] -= twoPi;
  while(y[1] <= -M_PI) y[1] += twoPi;
}