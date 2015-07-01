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

    virtual Eigen::MatrixXd getFunc(const Eigen::MatrixXd& X) =0;
    virtual Eigen::MatrixXd getGrad(const Eigen::MatrixXd& FX) =0;
};

#endif /* ACTIVATIONFUNCTION_H_ */
