#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {

 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  // store predicted trajectory
  vector<double>  mpc_x;
  vector<double>  mpc_y;

  // update previous actuators
  void Update(const vector<double>& prev_acc);
private:
  double _prev_a;
  double _prev_delta;  
};

#endif /* MPC_H */
