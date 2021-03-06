#include "MPC.h"
#include "poly_utils.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

#include "json.hpp"
using json = nlohmann::json;

using CppAD::AD;
using namespace Utils;

// TODO: Set the timestep length and duration
size_t N  = 10;
double dt = 0.05;
size_t latency_steps = 2;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// this is to store the index-offset for each variables
// each variable has N values except for delta & actuator, 
// we only have N-1 values (control input for each time-step)
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
  // Fitted polynomial coefficients
  Eigen::VectorXd _coeffs;
  double _ref_v;
public:  
  FG_eval(const Eigen::VectorXd& coeffs, 
          const double ref_v) 
          : _coeffs(coeffs)
          , _ref_v(ref_v)
  { 
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;
    
    // Reference State Cost
    // TODO: Define the cost related the reference state and
    // any anything you think may be beneficial.
    for(int t = 0; t < N; ++t) {
      // sum-square cte
      fg[0] += CppAD::pow(vars[cte_start+t], 2);

      // sum-square epsi
      fg[0] += CppAD::pow(vars[epsi_start+t], 2);

      // sum-square difference between current speed and reference speed
      fg[0] += CppAD::pow(vars[v_start+t] - _ref_v, 2);
    }

    // minimize the use of acctuators
    for(int t = 0; t < N - 1; ++t) {
      fg[0] += CppAD::pow(vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t], 2);
    }
    
    // minimize the change of acctuators
    const double delta_w = 500.;
    for(int t = 0; t < N - 2; ++t) {
      fg[0] += delta_w * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // ensure the smooth transition from previous step
    //fg[0] += 100 * CppAD::pow(vars[delta_start + latency_steps] - vars[delta_start + latency_steps-1], 2);
    
    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start]    = vars[x_start];
    fg[1 + y_start]    = vars[y_start];
    fg[1 + psi_start]  = vars[psi_start];
    fg[1 + v_start]    = vars[v_start];
    fg[1 + cte_start]  = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int t = 1; t < N; t++) {
      // state at time t
      AD<double> x1   = vars[x_start + t];
      AD<double> y1   = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1   = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1= vars[epsi_start + t];

      // state at time t - 1
      AD<double> x0   = vars[x_start + t - 1];
      AD<double> y0   = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0   = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0= vars[epsi_start + t - 1];

      // actuator at time t - 1
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0     = vars[a_start + t - 1];

      // reference line & heading
      AD<double> f0      = polyeval(_coeffs, x0);
      AD<double> psides0 = CppAD::atan(polydiff(_coeffs, x0)); 

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // NOTE: The use of `AD<double>` and use of `CppAD`!
      // This is also CppAD can compute derivatives and pass
      // these to the solver.

      // TODO: Setup the rest of the model constraints i.e Model Dynamics
      fg[1 + x_start + t]   = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t]   = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 - (v0 * delta0 * dt )/ Lf );
      fg[1 + v_start + t]   = v1 - (v0 + a0 * dt);

      // cross-track error cte0 = f(x0) - y0, cte1 = cte0 + v0 * sin(epsi0) * dt;
      fg[1 + cte_start + t]  = cte1 - ((f0 - y0) + v0 * CppAD::sin(epsi0) * dt);
      fg[1 + epsi_start + t] = epsi1 - (psi0 - psides0 - (v0 * delta0 * dt )/ Lf );
    }
  }
};

typedef CPPAD_TESTVECTOR(double) Dvector;

void solDump(double ref_v,
             const Eigen::VectorXd& state,
             const CppAD::ipopt::solve_result<Dvector>& solution) {
  vector<double> mpc_x(N, 0.);
  vector<double> mpc_y(N, 0.);
  vector<double> a(N-1, 0.);
  vector<double> delta(N-1, 0.);

  for(size_t i = 0; i < N; ++i) {
    mpc_x[i] = solution.x[x_start+i];
    mpc_y[i] = solution.x[y_start+i];

    if (i < N - 1) {
      a[i]     = solution.x[a_start+i];
      delta[i] = solution.x[delta_start+i];
    }
  }
  
  // dump solution for analyze
  json sol;
  sol["mpc_x"]     = mpc_x;
  sol["mpc_y"]     = mpc_y;
  sol["mpc_a"]     = a;
  sol["mpc_delta"] = delta;
  sol["N"]         = N;
  sol["dt"]        = dt;
  sol["ref_v"]     = ref_v;
  sol["cost"]      = solution.obj_value;
  sol["x"]         = state[0];
  sol["y"]         = state[1];
  sol["psi"]       = state[2];
  sol["v"]         = state[3];
  sol["cte"]       = state[4];
  sol["epsi"]      = state[5];
  cout << "sol-data " << sol << endl;
}

//
// MPC class definition implementation.
//
MPC::MPC(double ref_v, double lower_v, bool verbose) 
  : mpc_x(vector<double>(N, 0.))
  , mpc_y(vector<double>(N, 0.))
  , _ref_v(ref_v)
  , _lower_v(lower_v)
  , _prev_a(.1)
  , _prev_delta(.0)
  , _verbose(verbose)
{}

MPC::~MPC() {}

vector<double> MPC::Solve(const Eigen::VectorXd& state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;  

  // initial state
  double x    = state[0];
  double y    = state[1];
  double psi  = state[2];
  double v    = state[3];
  double cte  = state[4];
  double epsi = state[5];

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  // number of model variables:
  // state is 6-element vector and the actuators is 2-element vector
  size_t n_vars = N * 6 + (N - 1) * 2;
  // TODO: Set the number of constraints: initial constraint (6) + (N-1) kinematic constraint (6 * (N-1)) 
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  // to the max negative and positive values.
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] =  1.0e19;
  }

  // The upper and lower limits of delta are set to deg2rad(-25) and deg2rad(25)
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.292;//-0.436332;
    vars_upperbound[i] =  0.292;//0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] =  1.0;
  }

  // to handle latency, the first few actuators are fixed from previous step
  for(int i = delta_start; i < delta_start + latency_steps; ++i) {
    vars_lowerbound[i] = _prev_delta;
    vars_upperbound[i] = _prev_delta;
  }

  for(int i = a_start; i < a_start + latency_steps; ++i) {
    vars_lowerbound[i] = _prev_a;
    vars_upperbound[i] = _prev_a;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // initial state constraint
  constraints_lowerbound[x_start]    = x;
  constraints_lowerbound[y_start]    = y;
  constraints_lowerbound[psi_start]  = psi;
  constraints_lowerbound[v_start]    = v;
  constraints_lowerbound[cte_start]  = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start]    = x;
  constraints_upperbound[y_start]    = y;
  constraints_upperbound[psi_start]  = psi;
  constraints_upperbound[v_start]    = v;
  constraints_upperbound[cte_start]  = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  double dyn_ref_v = (abs(cte) > 1.0) ? _lower_v : _ref_v; // in case cte is big we should slow-down our car
  FG_eval fg_eval(coeffs, dyn_ref_v);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  // std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.

  // update optimized trajectory
  for(size_t i = 0; i < N; ++i) {
    mpc_x[i] = solution.x[x_start+i];
    mpc_y[i] = solution.x[y_start+i];
  }
  
  // dump solution for analyze
  if (_verbose)
    solDump(_ref_v, state, solution);

  // we take the average of next latency_steps as new control input (since it will be fixed in the next latency_steps)
  double new_delta = solution.x[delta_start + latency_steps];
  double new_acc   = solution.x[a_start + latency_steps];

  return {new_delta, new_acc};
}

void MPC::Update(const vector<double>& prev_acc) {
  _prev_delta = prev_acc[0];
  _prev_a     = prev_acc[1];
}

bool MPC::IsVerbose() const {
  return _verbose;
}
