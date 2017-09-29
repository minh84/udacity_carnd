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
  int n_x = 4; // state = (px, py, v, yaw)
  VectorXd rmse = VectorXd::Zero(n_x);

  if (estimations.empty() || estimations.size() != ground_truth.size()) {
    cout << "Inputs to RMSE are INVALID!!!\n";
    return rmse;
  }

  // RMSE error
  for (int i = 0; i < estimations.size(); ++i) {
    for (int j = 0; j < n_x; ++j) {
      double err = estimations[i][j] - ground_truth[i][j];
      rmse[j] += err*err;
    }
  }

  // compute mean
  for (int j = 0; j < n_x; ++j) {
    rmse[j] /= estimations.size();
    rmse[j] = sqrt(rmse[j]);
  }
  return rmse;
}