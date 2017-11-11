#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {

 public:
  MPC(double ref_v, double lower_v, bool verbose);

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(const Eigen::VectorXd& state, Eigen::VectorXd coeffs);

  // store predicted trajectory
  vector<double>  mpc_x;
  vector<double>  mpc_y;

  // update previous actuators
  void Update(const vector<double>& prev_acc);
  
  bool IsVerbose() const;
private:
  double _ref_v;
  double _lower_v;
  double _prev_a;
  double _prev_delta;  
  bool   _verbose;
};

#endif /* MPC_H */
