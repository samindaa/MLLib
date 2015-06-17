/*
 * LIBLBFGSOptimizer.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "LIBLBFGSOptimizer.h"

LIBLBFGSOptimizer::LIBLBFGSOptimizer() :
    parameters(nullptr), dataFunction(nullptr), costFunction(nullptr)
{
}

LIBLBFGSOptimizer::~LIBLBFGSOptimizer()
{
  if (parameters)
  {
    lbfgs_free(parameters);
    parameters = nullptr;
  }
}

void LIBLBFGSOptimizer::optimize(Eigen::VectorXd& theta, DataFunction* dataFunction,
    CostFunction* costFunction)
{
  lbfgs_parameter_t lbfgs_parameter = { 6, 1e-5/*epsilon*/, 0, 1e-5/*delta*/, 200/*max_iterations*/,
      LBFGS_LINESEARCH_DEFAULT, 40, 1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16, 0.0, 0, -1, };

  lbfgsfloatval_t fx;
  parameters = lbfgs_malloc(theta.size());
  this->dataFunction = dataFunction;
  this->costFunction = costFunction;

  if (!parameters)
  {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < theta.size(); ++i)
    parameters[i] = theta(i);

  /*
   Start the L-BFGS LIBLBFGSOptimizer; this will invoke the callback functions
   evaluate() and progress() when necessary.
   */
  int ret = lbfgs(theta.size(), parameters, &fx, _evaluate, _progress, this, &lbfgs_parameter);

  /* Report the result. */
  printf("L-BFGS LIBLBFGSOptimizer terminated with status code = %d\n", ret);
  printf("  fx = %f\n", fx);

  for (int i = 0; i < theta.size(); ++i)
    theta(i) = parameters[i];
}

lbfgsfloatval_t LIBLBFGSOptimizer::evaluate(const lbfgsfloatval_t *x, lbfgsfloatval_t *g,
    const int n, const lbfgsfloatval_t step)
{
  Eigen::VectorXd theta = Eigen::VectorXd::Map(x, n, 1);
  Eigen::VectorXd grad = costFunction->getGrad(theta, dataFunction->getTrainingX(),
      dataFunction->getTrainingY());
  for (int i = 0; i < theta.size(); ++i)
    g[i] = grad(i);  //<< fixme: find a better way
  return costFunction->getCost(theta, dataFunction->getTrainingX(), dataFunction->getTrainingY());
}

int LIBLBFGSOptimizer::progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls)
{
  printf("Iteration %d:\n", k);
  //Eigen::VectorXd theta = Eigen::VectorXd::Map(x, n, 1);
  //std::cout << theta.transpose() << std::endl;
  //  printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
  printf("  fx = %f \n", fx);
  printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("\n");
  return 0;
}

lbfgsfloatval_t LIBLBFGSOptimizer::_evaluate(void *instance, const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step)
{
  return reinterpret_cast<LIBLBFGSOptimizer*>(instance)->evaluate(x, g, n, step);
}

int LIBLBFGSOptimizer::_progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls)
{
  return reinterpret_cast<LIBLBFGSOptimizer*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n,
      k, ls);
}

