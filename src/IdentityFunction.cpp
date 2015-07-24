/*
 * IdentityFunction.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: sam
 */

#include "IdentityFunction.h"

Matrix_t IdentityFunction::getFunc(const Matrix_t& X)
{
  return X;
}

Matrix_t IdentityFunction::getGrad(const Matrix_t& FX)
{
  return Matrix_t::Identity(FX.rows(), FX.cols());
}



