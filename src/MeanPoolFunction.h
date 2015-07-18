/*
 * MeanPoolFunction.h
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#ifndef MEANPOOLFUNCTION_H_
#define MEANPOOLFUNCTION_H_

#include "PoolingFunction.h"

class MeanPoolFunction: public PoolingFunction
{
  public:
    MeanPoolFunction(const int& numFilters, const int& outputDim);
    virtual ~MeanPoolFunction();
    Poolings* pool(const Convolutions* convolutedFunctions, const int& poolDim);
    void delta_pool(const int& poolDim);
    void clear();
};

#endif /* MEANPOOLFUNCTION_H_ */
