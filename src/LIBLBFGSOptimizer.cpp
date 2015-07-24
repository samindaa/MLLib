/*
 * LIBLBFGSOptimizer.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "LIBLBFGSOptimizer.h"

LIBLBFGSOptimizer::LIBLBFGSOptimizer(const int& max_iterations) :
    max_iterations(max_iterations), parameters(nullptr), dataFunction(nullptr), //
    costFunction(nullptr)
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

void LIBLBFGSOptimizer::optimize(Vector_t& theta, DataFunction* dataFunction,
    CostFunction* costFunction)
{
  lbfgs_parameter_t lbfgs_parameter = { 6, 1e-5/*epsilon*/, 0, 1e-5/*delta*/,
      max_iterations/*max_iterations*/, LBFGS_LINESEARCH_DEFAULT, 40, 1e-20, 1e20, 1e-4, 0.9, 0.9,
      1.0e-16, 0.0, 0, -1, };

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
  printf("L-BFGS LIBLBFGSOptimizer terminated with status code = %s\n", toVerbose(ret).c_str());
  printf("  fx = %f\n", fx);

  for (int i = 0; i < theta.size(); ++i)
    theta(i) = parameters[i];
}

lbfgsfloatval_t LIBLBFGSOptimizer::evaluate(const lbfgsfloatval_t *x, lbfgsfloatval_t *g,
    const int n, const lbfgsfloatval_t step)
{
  Vector_t theta = Vector_t::Map(x, n, 1);
  Vector_t grad;
  double cost = costFunction->evaluate(theta, dataFunction->getTrainingX(),
      dataFunction->getTrainingY(), grad);

#pragma omp parallel for
  for (int i = 0; i < grad.size(); ++i)
    g[i] = grad(i);

  return cost;
}

int LIBLBFGSOptimizer::progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls)
{
  printf("Iteration %d:\n", k);
  //Vector_t theta = Vector_t::Map(x, n, 1);
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

std::string LIBLBFGSOptimizer::toVerbose(const int& ret)
{
  switch (ret)
  {
    case LBFGS_SUCCESS:
      return "L-BFGS reaches convergence";
    case LBFGS_STOP:
      return "LBFGS_STOP";
    case LBFGS_ALREADY_MINIMIZED:
      return "The initial variables already minimize the objective function";
    case LBFGSERR_UNKNOWNERROR:
      return "Unknown error";
    case LBFGSERR_LOGICERROR:
      return "Logic error";
    case LBFGSERR_OUTOFMEMORY:
      return "Insufficient memory";
    case LBFGSERR_CANCELED:
      return "The minimization process has been canceled";
    case LBFGSERR_INVALID_N:
      return "Invalid number of variables specified";
    case LBFGSERR_INVALID_N_SSE:
      return "Invalid number of variables (for SSE) specified";
    case LBFGSERR_INVALID_X_SSE:
      return "The array x must be aligned to 16 (for SSE)";
    case LBFGSERR_INVALID_EPSILON:
      return "Invalid parameter lbfgs_parameter_t::epsilon specified";
    case LBFGSERR_INVALID_TESTPERIOD:
      return "Invalid parameter lbfgs_parameter_t::past specified";
    case LBFGSERR_INVALID_DELTA:
      return "Invalid parameter lbfgs_parameter_t::delta specified";
    case LBFGSERR_INVALID_LINESEARCH:
      return "Invalid parameter lbfgs_parameter_t::linesearch specified";
    case LBFGSERR_INVALID_MINSTEP:
      return "Invalid parameter lbfgs_parameter_t::max_step specified";
    case LBFGSERR_INVALID_MAXSTEP:
      return "Invalid parameter lbfgs_parameter_t::max_step specified";
    case LBFGSERR_INVALID_FTOL:
      return "Invalid parameter lbfgs_parameter_t::ftol specified";
    case LBFGSERR_INVALID_WOLFE:
      return "Invalid parameter lbfgs_parameter_t::wolfe specified";
    case LBFGSERR_INVALID_GTOL:
      return "Invalid parameter lbfgs_parameter_t::gtol specified";
    case LBFGSERR_INVALID_XTOL:
      return "Invalid parameter lbfgs_parameter_t::xtol specified";
    case LBFGSERR_INVALID_MAXLINESEARCH:
      return "Invalid parameter lbfgs_parameter_t::max_linesearch specified";
    case LBFGSERR_INVALID_ORTHANTWISE:
      return "Invalid parameter lbfgs_parameter_t::orthantwise_c specified";
    case LBFGSERR_INVALID_ORTHANTWISE_START:
      return "Invalid parameter lbfgs_parameter_t::orthantwise_start specified";
    case LBFGSERR_INVALID_ORTHANTWISE_END:
      return "Invalid parameter lbfgs_parameter_t::orthantwise_end specified";
    case LBFGSERR_OUTOFINTERVAL:
      return "The line-search step went out of the interval of uncertainty";
    case LBFGSERR_INCORRECT_TMINMAX:
      return "A logic error occurred; alternatively, the interval of uncertainty became too small";
    case LBFGSERR_ROUNDING_ERROR:
      return "A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions";
    case LBFGSERR_MINIMUMSTEP:
      return "The line-search step became smaller than lbfgs_parameter_t::min_step";
    case LBFGSERR_MAXIMUMSTEP:
      return "The line-search step became larger than lbfgs_parameter_t::max_step";
    case LBFGSERR_MAXIMUMLINESEARCH:
      return "The line-search routine reaches the maximum number of evaluations";
    case LBFGSERR_MAXIMUMITERATION:
      return "The algorithm routine reaches the maximum number of iterations";
    case LBFGSERR_WIDTHTOOSMALL:
      return "Relative width of the interval of uncertainty is at most lbfgs_parameter_t::xtol";
    case LBFGSERR_INVALIDPARAMETERS:
      return "A logic error (negative line-search step) occurred";
    case LBFGSERR_INCREASEGRADIENT:
      return "The current search direction increases the objective function value";
    default:
      return "";
  }
}

