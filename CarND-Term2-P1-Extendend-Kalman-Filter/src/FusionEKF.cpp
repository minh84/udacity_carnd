#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() 
  : laser_measure_()
  , radar_measure_()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  MatrixXd R_laser = MatrixXd(2, 2);
  MatrixXd R_radar = MatrixXd(3, 3);

  //measurement covariance matrix - laser
  R_laser << 0.0225, 0,
             0, 0.0225;

  //measurement covariance matrix - radar
  R_radar << 0.09, 0,      0,
             0,    0.0009, 0,
             0,    0,      0.09;

  laser_measure_.setNoise(R_laser);
  radar_measure_.setNoise(R_radar);

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  // noise = 9.0
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
  This is a tweak to handle the case user click restart in the simulator
  Since we don't know how to capture this event from message between server/client (there is no special message for restart)
  but we know that when restart we have previous_timestamp_ > measurement_pack.timestamp_
  */
  
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1.0e6;
  bool  is_restarted = (dt < 0);

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_ || is_restarted) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    
    // first measurement: init Extendend Kalman Filter and reset all state

    cout << "EKF: " << endl;
    VectorXd x0 = VectorXd::Zero(4);
    MatrixXd P0 = MatrixXd::Identity(4, 4); 
    MatrixXd F  = MatrixXd::Identity(4, 4);
    MatrixXd Q  = MatrixXd::Zero(4, 4);
    P0(2, 2) = 100.;
    P0(3, 3) = 100.;

    // reset all initial states back to 0.
    laser_measure_.reset();
    radar_measure_.reset();

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.

      Note that measurement_pack.raw_measurements_ = (rho, phi, rhodot) so we need to convert back to px, py, vx, vy
      However, we can compute px, py but there are many (vx, vy) for a given rhodot, so we chose the projected one
      */
      x0 = radar_measure_.computeState0(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.

      For laser, measurement_pack.raw_measurements_ = (px, py) so we suppose vx, vy = 0
      */
      x0 = laser_measure_.computeState0(measurement_pack.raw_measurements_);
    }
    ekf_.Init(x0, P0, F, Q);

    // init first timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  previous_timestamp_ = measurement_pack.timestamp_;

  // update F, Q
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  double dt2 = dt * dt;
	  
  ekf_.Q_(2, 2) = dt2 * noise_ax_;
	ekf_.Q_(3, 3) = dt2 * noise_ay_;	
	ekf_.Q_(0, 2) = ekf_.Q_(2, 0) = dt * ekf_.Q_(2, 2) / 2.0;
	ekf_.Q_(1, 3) = ekf_.Q_(3, 1) = dt * ekf_.Q_(3, 3) / 2.0;
	ekf_.Q_(0, 0) = dt * ekf_.Q_(0, 2) / 2.0;
	ekf_.Q_(1, 1) = dt * ekf_.Q_(1, 3) / 2.0;

  // predict state
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.UpdateIEKF(measurement_pack.raw_measurements_, &radar_measure_, 5);
  } else {
    ekf_.Update(measurement_pack.raw_measurements_, &laser_measure_);
  }

  // print the output
  // cout << "x_ = " << ekf_.x_ << endl;
  // cout << "P_ = " << ekf_.P_ << endl;
}
