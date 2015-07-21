/*
 * LinearRegressionCostFunction.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#include "LinearCostFunction.h"
#include <iostream>

Vector_t LinearCostFunction::configure(const Matrix_t& X, const Matrix_t& Y)
{
  const int numberOfParameters = X.cols();
  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;
  Vector_t theta = (Vector_t::Random(numberOfParameters, 1).array() + 1.0f) * 0.5f * 0.001f;
  return theta;
}

double LinearCostFunction::evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y,
    Vector_t& grad)
{
  grad = X.transpose() * (X * theta - Y);
  return ((X * theta - Y).array().square().sum()) * 0.5f;
}

double LinearCostFunction::accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y)
{
  return sqrt((X * theta - Y).array().square().rowwise().sum().mean());
}

