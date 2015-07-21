/*
 * LogisticCostFunction.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "LogisticCostFunction.h"
#include <iostream>

LogisticCostFunction::LogisticCostFunction() :
    sigmoid(new SigmoidFunction())
{
}

LogisticCostFunction::~LogisticCostFunction()
{
  delete sigmoid;
}

Vector_t LogisticCostFunction::configure(const Matrix_t& X, const Matrix_t& Y)
{
  const int numberOfParameters = X.cols();
  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;
  Vector_t theta = (Vector_t::Random(numberOfParameters, 1).array() + 1.0f) * 0.5f
      * 0.0001f;
  return theta;
}

double LogisticCostFunction::evaluate(const Vector_t& theta, const Matrix_t& X,
    const Matrix_t& Y, Vector_t& grad)
{
  Matrix_t hx = sigmoid->getFunc(X * theta);
  grad = X.transpose() * (hx - Y);
  return -( //
  (Y.array() * hx.array().log()) + //
      ((1.0f - Y.array()) * (1.0f - hx.array()).log()) //
  ).sum();
}

double LogisticCostFunction::accuracy(const Vector_t& theta, const Matrix_t& X,
    const Matrix_t& Y)
{
  Matrix_t hx = sigmoid->getFunc(X * theta);
  int correct = 0;
  for (int i = 0; i < hx.rows(); ++i)
  {
    if ((hx(i, 0) > 0.5f && Y(i, 0) == 1) || (hx(i, 0) <= 0.5f && Y(i, 0) == 0))
      ++correct;
    else
      std::cout << "i: " << i << " pred: " << hx(i, 0) << " true: " << Y(i, 0) << std::endl;
  }
  return double(correct) * 100.0f / X.rows();
}
