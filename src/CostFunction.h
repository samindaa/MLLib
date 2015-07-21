/*
 * CostFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef COSTFUNCTION_H_
#define COSTFUNCTION_H_

#include "EigenFunction.h"

class CostFunction : public EigenFunction
{
  public:
    virtual ~CostFunction()
    {
    }

    virtual Vector_t configure(const Matrix_t& X, const Matrix_t& Y) =0;
    virtual double evaluate(const Vector_t& theta, const Matrix_t& X,
        const Matrix_t& Y, Vector_t& grad) =0;
    virtual double accuracy(const Vector_t& theta, const Matrix_t& X,
        const Matrix_t& Y) =0;

    double getNumGrad(const Vector_t& theta, const Matrix_t& X,
        const Matrix_t& Y);
    double getNumGrad(const Vector_t& theta, const Matrix_t& X,
        const Matrix_t& Y, const int& numChecks);
};

#endif /* COSTFUNCTION_H_ */
