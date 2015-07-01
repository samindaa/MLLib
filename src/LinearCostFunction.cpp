/*
 * LinearRegressionCostFunction.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#include "LinearCostFunction.h"
#include <iostream>

Eigen::VectorXd LinearCostFunction::configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
{
  const int numberOfParameters = X.cols();
  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;
  Eigen::VectorXd theta = (Eigen::VectorXd::Random(numberOfParameters, 1).array() + 1.0f) * 0.5f
      * 0.001f;
  return theta;
}

double LinearCostFunction::evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, Eigen::VectorXd& grad)
{
  grad = X.transpose() * (X * theta - Y);
  return ((X * theta - Y).array().square().sum()) * 0.5f;
}

/*
 Eigen::VectorXd LinearCostFunction::getGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
 const Eigen::MatrixXd& Y)
 {
 return X.transpose() * (X * theta - Y);
 }

 double LinearCostFunction::getCost(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
 const Eigen::MatrixXd& Y)
 {
 return ((X * theta - Y).array().square().sum()) * 0.5f;
 }
 */

double LinearCostFunction::accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  return sqrt((X * theta - Y).array().square().rowwise().sum().mean());
}

