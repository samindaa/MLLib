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
    Eigen::MatrixXd W;
    Eigen::MatrixXd GradientW;
    Eigen::VectorXd b;
    Eigen::VectorXd gradientb;
    Eigen::MatrixXd Delta;
    Eigen::MatrixXd Z; //<< Activation
    Eigen::MatrixXd A; //<< Input to the layer
    Function* function;

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
