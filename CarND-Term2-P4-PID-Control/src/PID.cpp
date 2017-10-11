#include "PID.h"
#include <algorithm>
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  // we need to use this since the parameters same name as member (TODO: fix this ambiguity)
  Kp_ = Kp;
  Ki_ = Ki;
  Kd_ = Kd;

  nb_update_ = 0;
  sum_sqr_err_ = 0.;
  p_error_ = 0.;
  d_error_ = 0.;
  i_error_ = 0.;
}

void PID::UpdateError(double cte) {
  ++nb_update_;
  p_error_ = cte;
  sum_sqr_err_ += cte*cte;
  d_error_ = (nb_update_ == 1) ? 0. : cte - prev_cte_;
  i_error_+= cte;
    
  prev_cte_ = cte;
}

double PID::TotalError() {
  double err = Kp_ * p_error_ + Kd_ * d_error_ + Ki_ * i_error_;
  return max(-1.0, min(1.0, err));
}

void PID::PrintDebug() const {
  std::cout << "\nUpdate step " << nb_update_ << "\n";
  std::cout << "Kp=" << Kp_ << ", Kd=" << Kd_ << ", Ki=" << Ki_ << "\n";
  std::cout << "p_err=" << p_error_ << ", d_err=" << d_error_ << ", i_err=" << i_error_ << "\n";
  std::cout << "mean_err=" << sum_sqr_err_ / nb_update_ << "\n";
}