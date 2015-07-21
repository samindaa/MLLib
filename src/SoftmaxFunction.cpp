/*
 * SoftmaxFunction.cpp
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#include "SoftmaxFunction.h"
#include <cassert>

Matrix_t SoftmaxFunction::getFunc(const Matrix_t& X)
{
  Matrix_t A = X;
  Vector_t rowMaxCoff = A.rowwise().maxCoeff();
  A.colwise() -= rowMaxCoff;
  Matrix_t MatExp = A.array().exp().matrix();
  Vector_t divider = MatExp.rowwise().sum().cwiseInverse();
  return divider.asDiagonal() * MatExp;
}

Matrix_t SoftmaxFunction::getGrad(const Matrix_t& FX)
{
  assert(false);
  return Matrix_t();
}

