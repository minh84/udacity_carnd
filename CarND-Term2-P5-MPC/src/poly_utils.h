#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include <cppad/cppad.hpp>

namespace Utils {
    // evaluate polynomial given coefficients and x
    double polyeval(const Eigen::VectorXd& coeffs, double x);

    // evaluate derivative of polynomial given coefficients and x
    double polydiff(const Eigen::VectorXd& coeffs, double x);

    // AD version
    CppAD::AD<double> polyeval(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x);
    CppAD::AD<double> polydiff(const Eigen::VectorXd& coeffs, const CppAD::AD<double>& x);

    // fit polynome to points
    // return coefficients
    Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);
}