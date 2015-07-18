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

Eigen::MatrixXd WhiteningFunction::gen(const Eigen::MatrixXd& X)
{
  Eigen::MatrixXd Xr = X;
  if (config->getValue("zeroMean", false)) // for each row: assuming the statistics of rows are similar
    Xr.colwise() -= Xr.rowwise().mean();

  if (config->getValue("pcaWhitening", false))
  {
    Eigen::MatrixXd Sigma = Xr.transpose() * Xr / double(Xr.rows());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Sigma, Eigen::ComputeFullU);
    const Eigen::MatrixXd& U = svd.matrixU();
    const Eigen::VectorXd& S = svd.singularValues();

    Xr = Xr * U * ((S.array() + //
        config->getValue("epsilon", 0.01f)).sqrt().inverse().matrix().asDiagonal());

    if (config->getValue("zcaWhitening", false))
      Xr = Xr * U.transpose();

  }

  return Xr;
}

void WhiteningFunction::zeroMean(Eigen::MatrixXd& X)
{
}

void WhiteningFunction::pcaWhitening(Eigen::MatrixXd& X)
{
}

void WhiteningFunction::zcaWhitening(Eigen::MatrixXd& X)
{
}

