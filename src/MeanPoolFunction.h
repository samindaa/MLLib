/*
 * MeanPoolFunction.h
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#ifndef MEANPOOLFUNCTION_H_
#define MEANPOOLFUNCTION_H_

#include "PoolFunction.h"

class MeanPoolFunction: public PoolFunction
{
  public:
    MeanPoolFunction(const int& numFilters, const int& outputDim);
    virtual ~MeanPoolFunction();
    PooledFunctions* pool(const ConvolutedFunctions* convolutedFunctions, const int& poolDim);
};

#endif /* MEANPOOLFUNCTION_H_ */
