#include <iostream>
#include "kalman_filter.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(const VectorXd &x_in, 
                        const MatrixXd &P_in, 
                        const MatrixXd &F_in, 
                        const MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
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

void KalmanFilter::Update(const VectorXd &z, MeasureModel* model) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y  = z - model->measure(x_);   //residual
 
  const MatrixXd& H = model->jacobian(x_);
  MatrixXd Ht = H.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S  = H * PHt + model->getNoise();
  MatrixXd K  = PHt * S.inverse();

  // update state & covar
  model->normalize(y); // e.g for Radar we want y[1] in range [-pi, pi]
  x_ += K * y;
  P_ -= K * H * P_;
}

void KalmanFilter::UpdateIEKF(const Eigen::VectorXd &z, MeasureModel* model, int iter) {
  VectorXd x_ki = x_;

  MatrixXd H, K;

  for (int i = 0; i < iter; ++i) {
    
    H = model->jacobian(x_ki);
    MatrixXd Ht = H.transpose();
    MatrixXd PHt = P_ * Ht;
    MatrixXd S  = H * PHt + model->getNoise();
    K  = PHt * S.inverse();
    VectorXd step = z - model->measure(x_ki) - H * (x_ - x_ki);
    model->normalize(step);
    x_ki = x_ + K * step;
  }

  x_ = x_ki;
  P_ -= K * H * P_;
}