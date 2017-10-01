#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_     = 3.;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4.;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.35;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.035;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.35;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  time_us_    = 0;                   // initial time = 0
  n_x_        = 5;                   // state = (px, py, v, phi, phidot) to represent CTRV model (constant turn rate and velocity magnitude model)
  n_aug_      = n_x_ + 2;            // process noise is 2d-vector (noise_a, noise_yawdd)
  n_sig_      = 2 * n_aug_ + 1;
  Xsig_pred_  = MatrixXd(n_x_, n_sig_);
  lambda_     = 3 - n_aug_;
  
  // set up weights
  weights_    = VectorXd(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  const double w1   = 0.5 / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_; ++i) {
    weights_(i) = w1;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  double delta_t = (meas_package.timestamp_ - time_us_) / 1.0e6; 

  if(!is_initialized_ || delta_t < 0.) {
    Init(meas_package);

    return;
  }

  Prediction(delta_t);

  // update time stamps
  time_us_ = meas_package.timestamp_;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    UpdateRadar(meas_package);
  }
}

void UKF::Init(const MeasurementPackage& meas_package) {
  // reset all variable 
  x_.fill(0);
  P_ = MatrixXd::Identity(n_x_, n_x_);
  Xsig_pred_.setZero();

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // Laser only gives px, py
    x_[0] = meas_package.raw_measurements_[0];
    x_[1] = meas_package.raw_measurements_[1];
  } else {
    // Radar only gives rho, phi, rhodot
    x_[0] = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
    x_[1] = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
    x_[2] = meas_package.raw_measurements_[2]; 
    x_[3] = meas_package.raw_measurements_[1];
    x_[4] = 0.;
  }

  is_initialized_ = true;
  time_us_ = meas_package.timestamp_;
}

void normalizeAngle(double& angle) {
  //angle normalization
  while (angle > M_PI) angle -= 2.*M_PI;
  while (angle <-M_PI) angle += 2.*M_PI;
}

void UKF::generateSigmaPoints(MatrixXd& Xsig_aug) const {
  // create augmented x
  VectorXd x_aug(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = x_aug(n_x_ + 1) = 0;

  // create matrix P
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_)     = std_a_ * std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();
  double scale = sqrt(lambda_ + n_aug_);

  // generate sigma-points
  Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1)          = x_aug + scale * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - scale * L.col(i);
  }
}

void UKF::predictSigmaPoints(MatrixXd& Xsig_pred, const MatrixXd& Xsig_aug,double delta_t) const {
  if (! (Xsig_pred.rows() == n_x_ && Xsig_pred_.cols() == n_sig_)) {
    Xsig_pred.resize(n_x_, n_sig_);
  }

  const double delta_t2 = delta_t * delta_t;
  // predict sigma-points i.e compute f(x_k, nu_k)
  const double EPS = 1e-3;
  for(int i = 0; i < n_sig_; ++i) {
    // define variables to make the code easier to understand
    double px      = Xsig_aug(0, i);
    double py      = Xsig_aug(1, i);
    double v       = Xsig_aug(2, i);
    double yaw     = Xsig_aug(3, i);
    double yawd    = Xsig_aug(4, i);
    double nua     = Xsig_aug(5, i);
    double nuyawdd = Xsig_aug(6, i);

    double yaw_p = yaw; // all others can be updated inplace
    yaw_p += yawd * delta_t;

    if (fabs(yawd) > EPS) {
      double v_over_yawd = v/yawd;
      px += v_over_yawd * ( sin(yaw_p) - sin(yaw));
      py += v_over_yawd * (-cos(yaw_p) + cos(yaw));
      
    } else {
      px += v * delta_t * cos(yaw);
      py += v * delta_t * sin(yaw);
    }
    
    // adding noise
    px += 0.5 * nua * delta_t2 * cos(yaw) ;
    py += 0.5 * nua * delta_t2 * sin(yaw) ;
    
    v        += delta_t * nua;
    yaw_p    += 0.5 * delta_t2 * nuyawdd;
    yawd     +=  delta_t * nuyawdd;

    
    //write predicted sigma point into right column
    Xsig_pred(0,i) = px;
    Xsig_pred(1,i) = py;
    Xsig_pred(2,i) = v;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd;
  }
}

void UKF::predictState(const MatrixXd& Xsig_pred, VectorXd* x_pred, MatrixXd* P_pred) const {
  if (x_pred) {
    *x_pred = Xsig_pred * weights_;
    if(P_pred) {
      P_pred->fill(0.);
      for (int i = 0; i < n_sig_; ++i) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        //angle normalization
        normalizeAngle(x_diff(3));

        (*P_pred) += weights_(i) * x_diff * x_diff.transpose() ;
      }
    }
  }  
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug;
  generateSigmaPoints(Xsig_aug);  
  
  predictSigmaPoints(Xsig_pred_, Xsig_aug, delta_t);

  // now we can predict x_, P_ (i.e compute x_k|k-1, P_k|k-1)
  predictState(Xsig_pred_, &x_, &P_);
}

void UKF::getPredictedState(VectorXd& x_pred, double delta_t) const {
 
  MatrixXd Xsig_aug;
  generateSigmaPoints(Xsig_aug);  
  
  MatrixXd Xsig_pred;
  predictSigmaPoints(Xsig_pred, Xsig_aug, delta_t);
  predictState(Xsig_pred, &x_pred, nullptr);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // sigma-point of Lidar measure
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  Zsig = Xsig_pred_.topLeftCorner(n_z, n_sig_); // lidar only returns (px, py)

  // prediction of Lidar measure
  VectorXd z_pred = Zsig * weights_;
  
  // compute variance & cross-variance
  MatrixXd S = MatrixXd::Zero(n_z,  n_z);
  MatrixXd T = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < n_sig_; ++i) {
    //residual
    VectorXd z_diff   = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // variance & cross-variance
    S += weights_(i) * z_diff * z_diff.transpose();
    T += weights_(i) * x_diff * z_diff.transpose();
  }

  // adding measurment noise
  S(0, 0) += std_laspx_*std_laspx_;
  S(1, 1) += std_laspy_*std_laspy_;
  
  // Kalman gain
  MatrixXd Sinv = S.inverse();
  MatrixXd K = T * Sinv;

  // Update state & covariance
  VectorXd residual =(meas_package.raw_measurements_ - z_pred); 
  x_ += K * residual;
  P_ -= T * K.transpose();  // K * S * K' = T * invS * S * K' = T * K'  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  const double RHO_EPS = 1e-9;
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
    
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //rho
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / max(RHO_EPS, Zsig(0, i));  //rho_dot (we handle the case rho = 0. by thresholding)
  }

  // prediction of Lidar measure
  VectorXd z_pred = Zsig * weights_;
  
  // compute variance & cross-variance
  MatrixXd S = MatrixXd::Zero(n_z,  n_z);
  MatrixXd T = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < n_sig_; ++i) {
    //residual
    VectorXd z_diff   = Zsig.col(i) - z_pred;
    
    // we need to normalize angle to be between [-pi, pi]
    normalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // variance & cross-variance
    S += weights_(i) * z_diff * z_diff.transpose();
    T += weights_(i) * x_diff * z_diff.transpose();
  }

  // adding measurment noise
  S(0, 0) += std_radr_   * std_radr_;
  S(1, 1) += std_radphi_ * std_radphi_;
  S(2, 2) += std_radrd_  * std_radrd_;
  
  // Kalman gain
  MatrixXd Sinv = S.inverse();
  MatrixXd K = T * Sinv;
  
  // compute and normalize residual
  VectorXd residual =(meas_package.raw_measurements_ - z_pred);  
  normalizeAngle(residual(1));

  // Update state & covariance
  x_ += K * residual;
  P_ -= T * K.transpose();  // K * S * K' = T * invS * S * K' = T * K'  
}
