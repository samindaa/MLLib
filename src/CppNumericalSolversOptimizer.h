/*
 * CppNumericalSolversOptimizer.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef CPPNUMERICALSOLVERSOPTIMIZER_H_
#define CPPNUMERICALSOLVERSOPTIMIZER_H_

#include "Optimizer.h"
#include "CppNumericalSolvers/LbfgsSolver.h"

class CppNumericalSolversOptimizer: public Optimizer
{
  private:
    pwie::LbfgsSolver lbfgs;

  public:
    CppNumericalSolversOptimizer();
    ~CppNumericalSolversOptimizer();
    void optimize(Eigen::VectorXd& theta, DataFunction* dataFunction, CostFunction* costFunction);
};

#endif /* CPPNUMERICALSOLVERSOPTIMIZER_H_ */
