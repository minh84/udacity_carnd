#ifndef MEASURE_MODEL_H_
#define MEASURE_MODEL_H_

#include "Eigen/Dense"

//!  MeasureModel base class
/*!
  This is to represent the measurement step:
      z(k) = h(x(k)) + v(k)
  where x(k) is the state, z(k) is the measurement and v(k) is the noise

  This class allows to compute
    h(x), dh(x)/dx
*/
class MeasureModel {
protected:
  Eigen::VectorXd hx_;    // h(x) a vector with shape (output_dim)
  Eigen::MatrixXd Hj_;    // Hj_ = dh(x) / dx with shape (output_dim, input_dim)
  Eigen::MatrixXd R_;     // R_ = covariance of v with shape (output_dim, output_dim)
public:
  /**
   * Constructor.
   */
  MeasureModel(int input_dim, int output_dim) 
    : hx_(Eigen::VectorXd(output_dim))
    , Hj_(Eigen::MatrixXd(output_dim, input_dim))
  {
    reset();
  }

  /**
   * Destructor.
   */
  virtual ~MeasureModel() 
  {
    hx_.setZero();
    Hj_.setZero();
  }

  void reset() {

  }

  void setNoise(const Eigen::MatrixXd& noise_covar) {
    R_ = noise_covar;
  }

  const Eigen::MatrixXd& getNoise() const {
    return R_;
  }

  /**
   * Compute h(x) in measurement-update step
   */
  virtual const Eigen::VectorXd& measure(const Eigen::VectorXd& x) = 0;

  /**
   * Compute Jacobian Hj dh(x)/dx
   */
  virtual const Eigen::MatrixXd& jacobian(const Eigen::VectorXd& x) = 0;

  /**
   * Compute init state x0 given a measurement z
   */
  virtual Eigen::VectorXd computeState0(const Eigen::VectorXd& z) const = 0;

  /**
   * Normalize residual e.g for Radar we want to ensure y[1] is in range [-pi, pi]
   */
  virtual void normalize(Eigen::VectorXd& y) const 
  {
  }
};

class LaserMeasure : public MeasureModel {
public:
  /**
   * Constructor
   */
  LaserMeasure();
  
  /**
   * Destructor.
   */
  virtual ~LaserMeasure() 
  {
  }

  /**
   * Compute h(x) in measurement-update step
   */
  virtual const Eigen::VectorXd& measure(const Eigen::VectorXd& x);

  /**
   * Compute Jacobian Hj dh(x)/dx
   */
  virtual const Eigen::MatrixXd& jacobian(const Eigen::VectorXd& x);

  /**
   * Compute initial state x0 given a measurement z
   */
   virtual Eigen::VectorXd computeState0(const Eigen::VectorXd& z) const;
};

class RadarMeasure : public MeasureModel {
public:
  /**
   * Constructor
   */
  RadarMeasure();
  
  /**
   * Destructor.
   */
  virtual ~RadarMeasure()
  {
  }

  /**
   * Compute h(x) in measurement-update step
   */
  virtual const Eigen::VectorXd& measure(const Eigen::VectorXd& x);

  /**
   * Compute Jacobian Hj dh(x)/dx
   */
  virtual const Eigen::MatrixXd& jacobian(const Eigen::VectorXd& x);

  /**
   * Compute init state x0 given a measurement z
   */
   virtual Eigen::VectorXd computeState0(const Eigen::VectorXd& z) const;

   virtual void normalize(Eigen::VectorXd& y) const;
};

#endif
