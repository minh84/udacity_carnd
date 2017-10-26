#include "poly_utils.h"
using CppAD::AD;

namespace Utils {
  // Evaluate a polynomial.
  double polyeval(const Eigen::VectorXd& coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
      result += coeffs[i] * pow(x, i);
    }
    return result;
  }

  // Evaluate a derivative of polynomial
  double polydiff(const Eigen::VectorXd& coeffs, double x) {
    double result = 0.0;
    for (int i = 1; i < coeffs.size(); i++) {
      result += coeffs[i] * pow(x, i - 1) * i;
    }
    return result;
  }

  CppAD::AD<double> polyeval(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x) {
    CppAD::AD<double> result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * CppAD::pow(x, i);;
    }
    return result;
  }

  CppAD::AD<double> polydiff(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x) {
    CppAD::AD<double> result = 0.0;
    for (int i = 1; i < coeffs.size(); i++) {
        result += i * coeffs[i] * CppAD::pow(x, i - 1);;
    }
    return result;
  }
  
  // Fit a polynomial.
  // Adapted from
  // https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
  Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                          int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);
  
    for (int i = 0; i < xvals.size(); i++) {
      A(i, 0) = 1.0;
    }
  
    for (int j = 0; j < xvals.size(); j++) {
      for (int i = 0; i < order; i++) {
        A(j, i + 1) = A(j, i) * xvals(j);
      }
    }
  
    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
  }
}