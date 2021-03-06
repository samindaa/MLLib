/*
 * MNISTSoftmaxCostFunction.h
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#ifndef SOFTMAXCOSTFUNCTION_H_
#define SOFTMAXCOSTFUNCTION_H_

#include "CostFunction.h"
#include "SoftmaxFunction.h"

class SoftmaxCostFunction: public CostFunction
{
  private:
    ActivationFunction* softmax;
    double LAMBDA;

  public:
    SoftmaxCostFunction(const double& LAMBDA = 0.0f);
    ~SoftmaxCostFunction();

    Vector_t configure(const Matrix_t& X, const Matrix_t& Y);
    double evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y, Vector_t& grad);
    double accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y);

  private:
    static double sample(double)
    {
      static std::random_device rd;
      static std::mt19937 gen(rd());
      std::normal_distribution<> d(0.0f, 1.0f);
      return d(gen);
    }
};

#endif /* SOFTMAXCOSTFUNCTION_H_ */
