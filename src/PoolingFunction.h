/*
 * PoolingFunction.h
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#ifndef POOLINGFUNCTION_H_
#define POOLINGFUNCTION_H_

#include <vector>

#include "Convolutions.h"
#include "Poolings.h"

class PoolingFunction
{
  protected:
    Poolings* poolings;
    int outputDim;

  public:
    PoolingFunction(const int& numFilters, const int& outputDim) :
        poolings(new Poolings(numFilters, outputDim)), outputDim(outputDim)
    {
    }

    virtual ~PoolingFunction()
    {
      delete poolings;
    }

    virtual Poolings* pool(const Convolutions* convolutedFunctions,
        const int& poolDim) =0;
    virtual void delta_pool(const int& poolDim)=0;

};

#endif /* POOLINGFUNCTION_H_ */
