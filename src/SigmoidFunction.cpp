/*
 * SigmoidFunction.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#include "SigmoidFunction.h"

Matrix_t SigmoidFunction::getFunc(const Matrix_t& X)
{
  return (1.0f + (-X).array().exp()).inverse().matrix();
}

Matrix_t SigmoidFunction::getGrad(const Matrix_t& FX)
{
  return FX.cwiseProduct((1.0f - FX.array()).matrix());
}
