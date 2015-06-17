/*
 * SigmoidFunction.h
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#ifndef SIGMOIDFUNCTION_H_
#define SIGMOIDFUNCTION_H_

#include "Function.h"

class SigmoidFunction: public Function
{
  public:
    Eigen::MatrixXd getFunc(const Eigen::MatrixXd& X);
    Eigen::MatrixXd getGrad(const Eigen::MatrixXd& FX);
};

#endif /* SIGMOIDFUNCTION_H_ */
