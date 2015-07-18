/*
 * CostFunction.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#include "CostFunction.h"
#include <iostream>
#include <iomanip>

double CostFunction::getNumGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  Eigen::VectorXd numGrad = Eigen::VectorXd::Zero(theta.size());
  const double epsilon = 1e-3;
  Eigen::VectorXd e = Eigen::VectorXd::Zero(theta.size());
  Eigen::VectorXd grad, grad_tmp;
  evaluate(theta, X, Y, grad);

  for (int i = 0; i < theta.size(); ++i)
  {
    e(i) = epsilon;
    numGrad(i) = (evaluate(theta + e, X, Y, grad_tmp) - evaluate(theta - e, X, Y, grad_tmp))
        / (2.0f * epsilon);
    e(i) = 0;

    double error = fabs(double(grad(i)) - numGrad(i));

    std::cout << i << " " << grad(i) << " " << numGrad(i) << " " << error << std::endl;

  }

  Eigen::VectorXd error = grad - numGrad;
  std::cout << "error: " << (error.array().abs().sum()) << std::endl;
  Eigen::MatrixXd disp(theta.size(), 3);
  disp.col(0) = grad;
  disp.col(1) = numGrad;
  disp.col(2) = error.array().abs();

  std::cout << std::fixed << disp << std::endl;

  return error.array().abs().sum();
}

double CostFunction::getNumGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, const int& numChecks)
{

  const double epsilon = 1e-3;
  double sumError = 0.0f;

  for (int i = 0; i < numChecks; ++i)
  {
    Eigen::VectorXd T = theta;
    int j = rand() % theta.size();

    Eigen::VectorXd T0 = T;
    Eigen::VectorXd T1 = T;

    T0(j) -= epsilon;
    T1(j) += epsilon;
    Eigen::VectorXd grad, grad_tmp;
    double cost = evaluate(T, X, Y, grad);

    double gradEst = (evaluate(T1, X, Y, grad_tmp) - evaluate(T0, X, Y, grad_tmp))
        / (2.0f * epsilon);
    double error = fabs(double(grad(j)) - gradEst);

    std::cout << std::left << std::setw(5) << i << std::setw(6) << j << std::setw(15) << error
        << std::setw(15) << grad(j) << std::setw(15) << gradEst << std::setw(15) << cost;
    std::cout << std::endl;

    sumError += error;
  }

  std::cout << "average: " << (sumError / numChecks) << std::endl;
  return sumError / numChecks;

}
