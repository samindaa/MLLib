/*
 * LogisticCostFunction.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "LogisticCostFunction.h"
#include <iostream>

LogisticCostFunction::LogisticCostFunction() :
    sigmoid(new SigmoidFunction())
{
}

LogisticCostFunction::~LogisticCostFunction()
{
  delete sigmoid;
}

Eigen::VectorXd LogisticCostFunction::configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
{
  const int numberOfParameters = X.cols();
  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;
  Eigen::VectorXd theta = (Eigen::VectorXd::Random(numberOfParameters, 1).array() + 1.0f) * 0.5f
      * 0.0001f;
  return theta;
}

Eigen::VectorXd LogisticCostFunction::getGrad(const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& y)
{
  return X.transpose() * (sigmoid->getFunc(X * theta) - y);
}

double LogisticCostFunction::getCost(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& y)
{
  Eigen::MatrixXd hx = sigmoid->getFunc(X * theta);
  return -( //
  (y.array() * hx.array().log()) + //
      ((1.0f - y.array()) * (1.0f - hx.array()).log()) //
  ).sum();
}

double LogisticCostFunction::accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  Eigen::MatrixXd hx = sigmoid->getFunc(X * theta);
  int correct = 0;
  for (int i = 0; i < hx.rows(); ++i)
  {
    if ((hx(i, 0) > 0.5f && Y(i, 0) == 1) || (hx(i, 0) <= 0.5f && Y(i, 0) == 0))
      ++correct;
    else
      std::cout << "i: " << i << " pred: " << hx(i, 0) << " true: " << Y(i, 0) << std::endl;
  }
  return double(correct) * 100.0f / X.rows();
}
