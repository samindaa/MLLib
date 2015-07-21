/*
 * SupervisedNeuralNetworkCostFunction.h
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#ifndef SUPERVISEDNEURALNETWORKCOSTFUNCTION_H_
#define SUPERVISEDNEURALNETWORKCOSTFUNCTION_H_

#include <vector>
#include "CostFunction.h"
#include "SigmoidFunction.h"
#include "SupervisedNeuralNetworkLayer.h"

class SupervisedNeuralNetworkCostFunction: public CostFunction
{
  private:
    Vector_t topology;
    std::vector<SupervisedNeuralNetworkLayer*> layers;
    int numberOfParameters;
    double LAMBDA;

  public:
    SupervisedNeuralNetworkCostFunction(const Vector_t& topology, const double& LAMBDA = 0.0f);
    ~SupervisedNeuralNetworkCostFunction();

    Vector_t configure(const Matrix_t& X, const Matrix_t& Y);
    double evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y, Vector_t& grad);

    double accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y);

  private:
    void toTheta(Vector_t& grad);
    void toGradient(Vector_t& grad);
    void toWeights(const Vector_t& theta);
    void forwardPass(const Matrix_t& X);
};

#endif /* SUPERVISEDNEURALNETWORKCOSTFUNCTION_H_ */
