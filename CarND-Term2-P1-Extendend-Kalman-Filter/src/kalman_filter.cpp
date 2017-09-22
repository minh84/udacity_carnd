#include "kalman_filter.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  
    Note that both Laser/Radar shares the same prediction since the state-transition 
    is the same independent of sensor input
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateStep(const VectorXd& y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S  = H_ * P_ * Ht + R_;
  MatrixXd K  = P_ * Ht * S.inverse();

  // update state & covar
  x_ += K * y;
  P_ -= K * H_ * P_; // (I - K*H)*P' = P' - K * H * P'
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y  = z - H_ * x_; //residual
  UpdateStep(y);
}

/**
* A helper method to convert from cartesian to polar
*/
VectorXd Cartesian2Polar(const VectorXd& cartesian) {
  /**
  The input cartesian is a vector of (px, py, x, y)
  */
  float rho = sqrt(cartesian[0] * cartesian[0] + cartesian[1] * cartesian[1]);
  float phi = atan2(cartesian[1], cartesian[0]);
  float rhodot = (cartesian[0] * cartesian[2] + cartesian[1] * cartesian[3]) / rho;
  VectorXd polar(3);
  polar << rho, phi, rhodot;
  return polar;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations

  Here is only difference between EKF vs KF
    1) y = z - h(x_) (instead of H_ * x_)

  */
  VectorXd y  = z - Cartesian2Polar(x_);

  // we need to ensure y[1] is in between -pi, pi: this step is crucial
  while(y[1] >= M_PI)  y[1] -= M_PI;
  while(y[1] <= -M_PI) y[1] += M_PI;

  UpdateStep(y);
}
