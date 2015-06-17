/*
 * SigmoidFunction.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#include "SigmoidFunction.h"

Eigen::MatrixXd SigmoidFunction::getFunc(const Eigen::MatrixXd& X)
{
  return (1.0f + (-X).array().exp()).inverse().matrix();
}

Eigen::MatrixXd SigmoidFunction::getGrad(const Eigen::MatrixXd& FX)
{
  return FX.cwiseProduct((1.0f - FX.array()).matrix());
}
