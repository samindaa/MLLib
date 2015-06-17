/*
 * SoftmaxFunction.cpp
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#include "SoftmaxFunction.h"
#include <cassert>

Eigen::MatrixXd SoftmaxFunction::getFunc(const Eigen::MatrixXd& X)
{
  Eigen::MatrixXd A = X;
  Eigen::VectorXd rowMaxCoff = A.rowwise().maxCoeff();
  A.colwise() -= rowMaxCoff;
  Eigen::MatrixXd MatExp = A.array().exp().matrix();
  Eigen::VectorXd divider = MatExp.rowwise().sum().cwiseInverse();
  return divider.asDiagonal() * MatExp;
}

Eigen::MatrixXd SoftmaxFunction::getGrad(const Eigen::MatrixXd& FX)
{
  assert(false);
  return Eigen::MatrixXd();
}

