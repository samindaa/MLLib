/*
 * MeanPoolFunction.cpp
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#include "MeanPoolFunction.h"
#include <cassert>

MeanPoolFunction::MeanPoolFunction(const int& numFilters, const int& outputDim) :
    PoolFunction(numFilters, outputDim)
{
}

MeanPoolFunction::~MeanPoolFunction()
{
  for (auto iter = pooledFunctions->pooling.begin(); iter != pooledFunctions->pooling.end(); ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      delete iter2->second;
  }
}

PooledFunctions* MeanPoolFunction::pool(const ConvolutedFunctions* convolutedFunctions,
    const int& poolDim)
{
  for (auto iter = convolutedFunctions->convolutions.begin();
      iter != convolutedFunctions->convolutions.end(); ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
    {
      ConvolutedFunction* convolutedFunction = iter2->second;

      assert(convolutedFunction->X.rows() % poolDim == 0);
      assert(convolutedFunction->X.cols() % poolDim == 0);

      PooledFunction* pooledFunction = nullptr;
      auto iterPooling = pooledFunctions->pooling[iter->first].find(iter2->first);
      if (iterPooling != pooledFunctions->pooling[iter->first].end())
        pooledFunction = iterPooling->second;
      else
      {
        pooledFunction = new PooledFunction;
        pooledFunction->X.setZero(convolutedFunction->X.rows() / poolDim,
            convolutedFunction->X.cols() / poolDim);
        pooledFunctions->pooling[iter->first].insert(std::make_pair(iter2->first, pooledFunction));
      }

      int row = 0;
      int col = 0;
      for (int i = 0; i < convolutedFunction->X.rows(); i += poolDim)
      {
        for (int j = 0; j < convolutedFunction->X.cols(); j += poolDim)
        {
          pooledFunction->X(row, col) =
              convolutedFunction->X.block(i, j, poolDim, poolDim).array().mean();
          ++col;
        }
        assert(col == pooledFunction->X.cols());
        col = 0;
        ++row;
      }
      assert(row == pooledFunction->X.rows());
    }
  }

  return pooledFunctions;
}

