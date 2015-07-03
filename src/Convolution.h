/*
 * Convolution.h
 *
 *  Created on: Jun 27, 2015
 *      Author: sam
 */

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include "EigenFunction.h"

class Convolution: public EigenFunction
{
  public:
    Eigen::MatrixXd X;
};

#endif /* CONVOLUTION_H_ */
