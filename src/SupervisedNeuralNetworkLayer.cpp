/*
 * SupervisedNeuralNetworkLayer.cpp
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#include "SupervisedNeuralNetworkLayer.h"

SupervisedNeuralNetworkLayer::SupervisedNeuralNetworkLayer(const int& size,
    const FunctionType& functionType) :
    size(size), function(nullptr)
{
  if (functionType == SIGMOD_FUNCTION)
    function = new SigmoidFunction;
  else if (functionType == SOFTMAX_FUNCTION)
    function = new SoftmaxFunction;
}

SupervisedNeuralNetworkLayer::~SupervisedNeuralNetworkLayer()
{
  if (function)
  {
    delete function;
    function = nullptr;
  }
}

