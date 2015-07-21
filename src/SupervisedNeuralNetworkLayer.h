/*
 * SupervisedNeuralNetworkLayer.h
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#ifndef SUPERVISEDNEURALNETWORKLAYER_H_
#define SUPERVISEDNEURALNETWORKLAYER_H_

#include "Eigen/Dense"
#include "SigmoidFunction.h"
#include "SoftmaxFunction.h"

class SupervisedNeuralNetworkLayer
{
  public:
    int size;
    Matrix_t W;
    Matrix_t GradientW;
    Eigen::VectorXd b;
    Eigen::VectorXd gradientb;
    Matrix_t Delta;
    Matrix_t Z; //<< Activation
    Matrix_t A; //<< Input to the layer
    ActivationFunction* function;

    enum FunctionType
    {
      NULL_FUNCTION, //
      SIGMOD_FUNCTION, //
      SOFTMAX_FUNCTION
    };

    SupervisedNeuralNetworkLayer(const int& size, const FunctionType& functionType);
    ~SupervisedNeuralNetworkLayer();
};

#endif /* SUPERVISEDNEURALNETWORKLAYER_H_ */
