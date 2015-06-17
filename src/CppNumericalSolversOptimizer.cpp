/*
 * CppNumericalSolversOptimizer.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "CppNumericalSolversOptimizer.h"

CppNumericalSolversOptimizer::CppNumericalSolversOptimizer()
{
}

CppNumericalSolversOptimizer::~CppNumericalSolversOptimizer()
{
}

void CppNumericalSolversOptimizer::optimize(Eigen::VectorXd& theta, DataFunction* dataFunction,
    CostFunction* costFunction)
{
  DataFunction* df = dataFunction;
  CostFunction* cf = costFunction;

  // least squares
  auto objectiveFunction = [&df, &cf](const Eigen::VectorXd& x) -> double
  {
    double cost = cf->getCost(x, df->getTrainingX(),df->getTrainingY());
    std::cout << "fx: " << cost << std::endl;
    return cost;
  };

  // create derivative of function
  auto partialDerivatives = [&df, &cf](const Eigen::VectorXd& x, Eigen::VectorXd & grad) -> void
  {
    grad = cf->getGrad(x, df->getTrainingX(),df->getTrainingY());
    std::cout << "f_norm: " << x.norm() << " g_norm: " << grad.norm() << std::endl;
  };

  lbfgs.settings.maxIter = 200; //<< maxiter

  lbfgs.solve(theta, objectiveFunction, partialDerivatives);
}
