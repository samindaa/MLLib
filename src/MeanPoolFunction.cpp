/*
 * MeanPoolFunction.cpp
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#include "MeanPoolFunction.h"
#include <cassert>
#include <iostream>

MeanPoolFunction::MeanPoolFunction(const int& numFilters, const int& outputDim) :
    PoolingFunction(numFilters, outputDim)
{
}

MeanPoolFunction::~MeanPoolFunction()
{
  for (auto iter = poolings->unordered_map.begin(); iter != poolings->unordered_map.end(); ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      delete iter2->second;
  }
}

Poolings* MeanPoolFunction::pool(const Convolutions* convolutedFunctions, const int& poolDim)
{
  for (auto iter = convolutedFunctions->unordered_map.begin();
      iter != convolutedFunctions->unordered_map.end(); ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
    {
      Convolution* convolutedFunction = iter2->second;

      assert(convolutedFunction->X.rows() % poolDim == 0);
      assert(convolutedFunction->X.cols() % poolDim == 0);

      Pooling* pooling = nullptr;
      auto poolingIter = poolings->unordered_map[iter->first].find(iter2->first);
      if (poolingIter != poolings->unordered_map[iter->first].end())
        pooling = poolingIter->second;
      else
      {
        pooling = new Pooling;
        pooling->X.setZero(convolutedFunction->X.rows() / poolDim,
            convolutedFunction->X.cols() / poolDim);
        poolings->unordered_map[iter->first].insert(std::make_pair(iter2->first, pooling));
      }

      int row = 0;
      int col = 0;
      for (int i = 0; i < convolutedFunction->X.rows(); i += poolDim)
      {
        for (int j = 0; j < convolutedFunction->X.cols(); j += poolDim)
        {
          pooling->X(row, col) = convolutedFunction->X.block(i, j, poolDim, poolDim).array().mean();
          ++col;
        }
        assert(col == pooling->X.cols());
        col = 0;
        ++row;
      }
      assert(row == pooling->X.rows());
    }
  }

  return poolings;
}

void MeanPoolFunction::delta_pool(const int& poolDim)
{

  Eigen::MatrixXd Ones = Eigen::MatrixXd::Ones(poolDim, poolDim);
  Eigen::MatrixXd Tmp(outputDim * poolDim, outputDim * poolDim);
  const double fac = 1.0f / std::pow(poolDim, 2);

  for (auto iter = poolings->unordered_map_sensitivities.begin();
      iter != poolings->unordered_map_sensitivities.end(); ++iter)
  {

    //std::cout << "iter: " << iter->first << std::endl;

    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
    {

      Eigen::KroneckerProduct<Eigen::MatrixXd, Eigen::MatrixXd> Kron(iter2->second->X, Ones);
      Kron.evalTo(Tmp);
      iter2->second->X = Tmp;
      iter2->second->X.array() *= fac;

      //std::cout << "\t iter2: " << iter2->first << " => " << iter2->second->X.rows() << " x "
      //    << iter2->second->X.cols() << std::endl;

      //std::cout << iter2->second->X << std::endl;
      //std::cout << "^^^" << std::endl;
    }
  }
}

