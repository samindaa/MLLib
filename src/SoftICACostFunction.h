/*
 * SoftICACostFunction.h
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#ifndef SOFTICACOSTFUNCTION_H_
#define SOFTICACOSTFUNCTION_H_

#include "CostFunction.h"
#include <random>
#include <functional>

class SoftICACostFunction: public CostFunction
{
  private:
    int numFeatures; // number of filter banks to learn
    double lambda; // sparsity cost
    double epsilon; // epsilon to use in square-sqrt nonlinearity
    Matrix_t W;
    Matrix_t GradW;

  public:
    SoftICACostFunction(const int& numFeatures, const double& lambda, const double& epsilon);
    ~SoftICACostFunction();

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

#endif /* SOFTICACOSTFUNCTION_H_ */
