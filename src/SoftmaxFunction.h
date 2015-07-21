/*
 * SoftmaxFunction.h
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#ifndef SOFTMAXFUNCTION_H_
#define SOFTMAXFUNCTION_H_

#include "ActivationFunction.h"

class SoftmaxFunction: public ActivationFunction
{
  public:
    Matrix_t getFunc(const Matrix_t& X);
    Matrix_t getGrad(const Matrix_t& FX);
};

#endif /* SOFTMAXFUNCTION_H_ */
