/*
 * SigmoidFunction.h
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#ifndef SIGMOIDFUNCTION_H_
#define SIGMOIDFUNCTION_H_

#include "ActivationFunction.h"

class SigmoidFunction: public ActivationFunction
{
  public:
    Matrix_t getFunc(const Matrix_t& X);
    Matrix_t getGrad(const Matrix_t& FX);
};

#endif /* SIGMOIDFUNCTION_H_ */
