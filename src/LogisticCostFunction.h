/*
 * LogisticCostFunction.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef MNISTCOSTFUNCTION_H_
#define MNISTCOSTFUNCTION_H_

#include "CostFunction.h"
#include "SigmoidFunction.h"

class LogisticCostFunction: public CostFunction
{
  private:
    ActivationFunction* sigmoid;

  public:
    LogisticCostFunction();
    ~LogisticCostFunction();

    Vector_t configure(const Matrix_t& X, const Matrix_t& Y);
    double evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y, Vector_t& grad);

    double accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y);

};

#endif /* MNISTCOSTFUNCTION_H_ */
