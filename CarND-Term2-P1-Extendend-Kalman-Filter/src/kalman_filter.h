#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"
#include "measure_model.h"

class FusionEKF;

class KalmanFilter {
private:
  // state vector
  Eigen::VectorXd x_;
  
  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;
  
  friend class FusionEKF; // so that we can modify F_ & Q_ since F_ & Q_ depend on dt (and this's given at FusionEKF)

public:
  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param Q_in Process covariance matrix
   */
  void Init(const Eigen::VectorXd &x_in, 
            const Eigen::MatrixXd &P_in, 
            const Eigen::MatrixXd &F_in, 
            const Eigen::MatrixXd &Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict();

  /**
   * Updates the state for Extended Kalman Filter equations
   * @param z The measurement at k+1
   * @param model The measure model z_k = h(x_k) + v_k
   */
  void Update(const Eigen::VectorXd &z, MeasureModel* model);
  
  /**
   * Updates the state for Extended Kalman Filter equations using Iterative method
   * @param z The measurement at k+1
   * @param model The measure model z_k = h(x_k) + v_k
   */
   void UpdateIEKF(const Eigen::VectorXd &z, MeasureModel* model, int iter);

  /**
   * Get current state
   */
  const Eigen::VectorXd& State() const {
    return x_;
  }
};

#endif /* KALMAN_FILTER_H_ */
