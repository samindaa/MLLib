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
#include "Convolutions.h"

class ConvolutionFunction: public EigenFunction
{
  protected:
    FilterFunction* filterFunction;
    ActivationFunction* activationFunction;
    Convolutions* convolutions;

  public:
    ConvolutionFunction(FilterFunction* filterFunction, ActivationFunction* activationFunction);
    ~ConvolutionFunction();
    Convolutions* conv(const Eigen::MatrixXd& X, const Eigen::Vector2i& cofig);
    void validConv(Eigen::MatrixXd& Conv, const Eigen::MatrixXd& I, const Eigen::MatrixXd& W,
        const double& b);
};

#endif /* CONVOLUTIONFUNCTION_H_ */
