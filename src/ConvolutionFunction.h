/*
 * ConvolutionFunction.h
 *
 *  Created on: Jun 27, 2015
 *      Author: sam
 */

#ifndef CONVOLUTIONFUNCTION_H_
#define CONVOLUTIONFUNCTION_H_

#include "EigenFunction.h"
#include "FilterFunction.h"
#include "ActivationFunction.h"
#include "ConvolutedFunctions.h"

class ConvolutionFunction: public EigenFunction
{
  protected:
    FilterFunction* filterFunction;
    ActivationFunction* activationFunction;
    ConvolutedFunctions* convolutedFunctions;

  public:
    ConvolutionFunction(FilterFunction* filterFunction, ActivationFunction* activationFunction);
    ~ConvolutionFunction();
    ConvolutedFunctions* conv(const Eigen::MatrixXd& X, const Eigen::Vector2i& cofig);
};

#endif /* CONVOLUTIONFUNCTION_H_ */
