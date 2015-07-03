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
    convolutions(new Convolutions())
{
}

ConvolutionFunction::~ConvolutionFunction()
{
  for (auto iter = convolutions->unordered_map.begin(); iter != convolutions->unordered_map.end();
      ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      delete iter2->second;
  }
}

Convolutions* ConvolutionFunction::conv(const Eigen::MatrixXd& X, const Eigen::Vector2i& config)
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

      //const int limitRows = I.rows() - Wj.rows() + 1;
      //const int limitCols = I.cols() - Wj.cols() + 1;

      //Eigen::MatrixXd X_tmp(limitRows, limitCols);
      Eigen::MatrixXd X_tmp;
      validConv(X_tmp, I, Wj, bj);

      /*for (int row = 0; row < limitRows; ++row)
       {
       for (int col = 0; col < limitCols; ++col)
       {
       Eigen::MatrixXd Patch = I.block(row, col, Wj.rows(), Wj.cols());
       X_tmp(row, col) = Patch.cwiseProduct(Wj).sum() + bj;
       }
       }*/

      Convolution* convolution = nullptr;

#pragma omp critical
      {
        auto iter = convolutions->unordered_map[i].find(j);
        if (iter != convolutions->unordered_map[i].end())
          convolution = iter->second;
        else
        {
          convolution = new Convolution;
          convolutions->unordered_map[i].insert(std::make_pair(j, convolution));
        }
      }

      convolution->X = activationFunction->getFunc(X_tmp);

    }
  }

  return convolutions;
}

void ConvolutionFunction::validConv(Eigen::MatrixXd& Conv, const Eigen::MatrixXd& I,
    const Eigen::MatrixXd& W, const double& b)
{
  const int limitRows = I.rows() - W.rows() + 1;
  const int limitCols = I.cols() - W.cols() + 1;

  Conv.setZero(limitRows, limitCols);

  for (int row = 0; row < limitRows; ++row)
  {
    for (int col = 0; col < limitCols; ++col)
    {
      Eigen::MatrixXd Patch = I.block(row, col, W.rows(), W.cols());
      Conv(row, col) = Patch.cwiseProduct(W).sum() + b;
    }
  }
}

