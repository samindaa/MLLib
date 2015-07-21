/*
 * ActivationFunction.h
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_

#include "EigenFunction.h"

class ActivationFunction: public EigenFunction
{
  public:
    virtual ~ActivationFunction()
    {
    }

    virtual Matrix_t getFunc(const Matrix_t& X) =0;
    virtual Matrix_t getGrad(const Matrix_t& FX) =0;
};

#endif /* ACTIVATIONFUNCTION_H_ */
