/*
 * SoftmaxFunction.h
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#ifndef SOFTMAXFUNCTION_H_
#define SOFTMAXFUNCTION_H_

#include "Function.h"

class SoftmaxFunction: public Function
{
  public:
    Eigen::MatrixXd getFunc(const Eigen::MatrixXd& X);
    Eigen::MatrixXd getGrad(const Eigen::MatrixXd& FX);
};

#endif /* SOFTMAXFUNCTION_H_ */
