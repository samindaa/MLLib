/*
 * ConvolutedFunction.h
 *
 *  Created on: Jun 27, 2015
 *      Author: sam
 */

#ifndef CONVOLUTEDFUNCTION_H_
#define CONVOLUTEDFUNCTION_H_

#include "EigenFunction.h"

class ConvolutedFunction: public EigenFunction
{
  public:
    Eigen::MatrixXd X;
};

#endif /* CONVOLUTEDFUNCTION_H_ */
