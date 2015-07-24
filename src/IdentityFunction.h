/*
 * IdentityFunction.h
 *
 *  Created on: Jul 23, 2015
 *      Author: sam
 */

#ifndef IDENTITYFUNCTION_H_
#define IDENTITYFUNCTION_H_

#include "ActivationFunction.h"

class IdentityFunction: public ActivationFunction
{
  public:
    Matrix_t getFunc(const Matrix_t& X);
    Matrix_t getGrad(const Matrix_t& FX);
};

#endif /* IDENTITYFUNCTION_H_ */
