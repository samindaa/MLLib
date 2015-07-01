/*
 * ConvolutionFunction.cpp
 *
 *  Created on: Jun 27, 2015
 *      Author: sam
 */

#include "ConvolutionFunction.h"

ConvolutionFunction::ConvolutionFunction(FilterFunction* filterFunction,
    ActivationFunction* activationFunction) :
    filterFunction(filterFunction), activationFunction(activationFunction), //
    convolutedFunctions(new ConvolutedFunctions())
{
}

ConvolutionFunction::~ConvolutionFunction()
{
  for (auto iter = convolutedFunctions->convolutions.begin();
      iter != convolutedFunctions->convolutions.end(); ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      delete iter2->second;
  }
}

ConvolutedFunctions* ConvolutionFunction::conv(const Eigen::MatrixXd& X,
    const Eigen::Vector2i& config)
{
#pragma omp parallel for
  for (int i = 0; i < X.rows(); ++i)
  {
    Eigen::VectorXd x = X.row(i);
    Eigen::Map<Eigen::MatrixXd> I(x.data(), config(0), config(1));
    for (int j = 0; j < filterFunction->getWeights().cols(); ++j)
    {
      Eigen::VectorXd wj = filterFunction->getWeights().col(j);
      const double bj = filterFunction->getBiases()(j);

      // Filter
      Eigen::Map<Eigen::MatrixXd> Wj(wj.data(), filterFunction->getConfig()(0),
          filterFunction->getConfig()(1));

      // Do the valid convolution

      const int limitRows = I.rows() - Wj.rows() + 1;
      const int limitCols = I.cols() - Wj.cols() + 1;

      Eigen::MatrixXd X_tmp(limitRows, limitCols);

      for (int row = 0; row < limitRows; ++row)
      {
        for (int col = 0; col < limitCols; ++col)
        {
          Eigen::MatrixXd Patch = I.block(row, col, Wj.rows(), Wj.cols());
          X_tmp(row, col) = Patch.cwiseProduct(Wj).sum() + bj;
        }
      }

      ConvolutedFunction* convolutedFunction = nullptr;

#pragma omp critical
      {
        auto iter = convolutedFunctions->convolutions[i].find(j);
        if (iter != convolutedFunctions->convolutions[i].end())
          convolutedFunction = iter->second;
        else
        {
          convolutedFunction = new ConvolutedFunction;
          convolutedFunctions->convolutions[i].insert(std::make_pair(j, convolutedFunction));
        }
      }

      convolutedFunction->X = activationFunction->getFunc(X_tmp);

    }
  }

  return convolutedFunctions;
}

