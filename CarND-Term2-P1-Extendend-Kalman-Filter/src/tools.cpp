#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.empty() || estimations.size() != ground_truth.size()) {
    cout << "Inputs to RMSE are INVALID!!!\n";
    return rmse;
  }

  // RMSE error
  for (int i = 0; i < estimations.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      float err = estimations[i][j] - ground_truth[i][j];
      rmse[j] += err*err;
    }
  }

  // compute mean
  for (int j = 0; j < 4; ++j) {
    rmse[j] /= estimations.size();
    rmse[j] = sqrt(rmse[j]);
  }
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
    x_state = (px, py, vx, vy)
    polar = (rho, phi, rhodot)

    We want to compute dpolar / dx_state
  */
  const float eps = 1e-6;

  MatrixXd Hj(3, 4);
  float rho2 = x_state[0]*x_state[0] + x_state[1]*x_state[1];  
  if (rho2 > eps) {
    float rho  = sqrt(rho2);
    float rho3 = rho * rho2;
    float cross = x_state[2]*x_state[1] - x_state[3]*x_state[0];

    Hj << x_state[0]/rho,   x_state[1]/rho,  0, 0,
          -x_state[1]/rho2, x_state[0]/rho2, 0, 0,
          x_state[1] * cross/rho3, -x_state[0] * cross/rho3, x_state[0]/rho, x_state[1]/rho;
  } else {
    cout << "collision\n";
  }
  return Hj;
}

// VectorXd Tools::Cartesian2Polar(const VectorXd& cartesian) {
//   /**
//    The input cartesian is a vector of (px, py, x, y)
//   */
//   float rho = sqrt(cartesian[0] * cartesian[0] + cartesian[1] * cartesian[1]);
//   float phi = atan2(cartesian[1], cartesian[0]);
//   float rhodot = (cartesian[0] * cartesian[2] + cartesian[1] * cartesian[3]) / rho;
//   VectorXd polar(3);
//   polar << rho, phi, rhodot;
//   return polar;
// }
