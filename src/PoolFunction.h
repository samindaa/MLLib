/*
 * PoolFunction.h
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#ifndef POOLFUNCTION_H_
#define POOLFUNCTION_H_

#include <vector>
#include "ConvolutedFunctions.h"
#include "PooledFunctions.h"

class PoolFunction
{
  protected:
    PooledFunctions* pooledFunctions;

  public:
    PoolFunction(const int& numFilters, const int& outputDim) :
        pooledFunctions(new PooledFunctions(numFilters, outputDim))
    {
    }

    virtual ~PoolFunction()
    {
      delete pooledFunctions;
    }

    virtual PooledFunctions* pool(const ConvolutedFunctions* convolutedFunctions,
        const int& poolDim) =0;

};

#endif /* POOLFUNCTION_H_ */
