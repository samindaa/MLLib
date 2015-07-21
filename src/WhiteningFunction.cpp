/*
 * WhiteningFunction.cpp
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#include "WhiteningFunction.h"

WhiteningFunction::WhiteningFunction(Config* config) :
    config(config)
{
}

WhiteningFunction::~WhiteningFunction()
{
}

Matrix_t WhiteningFunction::gen(const Matrix_t& X)
{
  Matrix_t Xr = X;
  if (config->getValue("zeroMean", false)) // for each row: assuming the statistics of rows are similar
    Xr.colwise() -= Xr.rowwise().mean();

  if (config->getValue("pcaWhitening", false))
  {
    Matrix_t Sigma = Xr.transpose() * Xr / double(Xr.rows());
    Eigen::JacobiSVD<Matrix_t> svd(Sigma, Eigen::ComputeFullU);
    const Matrix_t& U = svd.matrixU();
    const Vector_t& S = svd.singularValues();

    Xr = Xr * U * ((S.array() + //
        config->getValue("epsilon", 0.01f)).sqrt().inverse().matrix().asDiagonal());

    if (config->getValue("zcaWhitening", false))
      Xr = Xr * U.transpose();

  }

  return Xr;
}

void WhiteningFunction::zeroMean(Matrix_t& X)
{
}

void WhiteningFunction::pcaWhitening(Matrix_t& X)
{
}

void WhiteningFunction::zcaWhitening(Matrix_t& X)
{
}

