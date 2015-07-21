/*
 * LinearCostFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef LINEARCOSTFUNCTION_H_
#define LINEARCOSTFUNCTION_H_

#include "CostFunction.h"

class LinearCostFunction: public CostFunction
{
  public:
    Vector_t configure(const Matrix_t& X, const Matrix_t& Y);
    double evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y, Vector_t& grad);
    double accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y);
};

#endif /* LINEARCOSTFUNCTION_H_ */
