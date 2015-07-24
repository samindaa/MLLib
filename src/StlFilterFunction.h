/*
 * StlFilterFunction.h
 *
 *  Created on: Jul 21, 2015
 *      Author: sam
 */

#ifndef STLFILTERFUNCTION_H_
#define STLFILTERFUNCTION_H_

#include "FilterFunction.h"

class StlFilterFunction: public FilterFunction
{
  private:

  public:
    StlFilterFunction(const int& filterDim, const Matrix_t& Wrica)
    {
      config << filterDim, filterDim;
      Weights = Wrica.transpose();
      biases.setZero(Weights.cols());
    }

    void configure()
    { // do nothing
    }
};

#endif /* STLFILTERFUNCTION_H_ */
