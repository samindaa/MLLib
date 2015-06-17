/*
 * LIBLBFGSOptimizer.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef LIBLBFGSOPTIMIZER_H_
#define LIBLBFGSOPTIMIZER_H_

#include "Optimizer.h"

class LIBLBFGSOptimizer: public Optimizer
{
  protected:
    lbfgsfloatval_t* parameters;
    DataFunction* dataFunction;
    CostFunction* costFunction;

  public:
    LIBLBFGSOptimizer();
    ~LIBLBFGSOptimizer();
    void optimize(Eigen::VectorXd& theta, DataFunction* dataFunction, CostFunction* costFunction);

  private:
    lbfgsfloatval_t evaluate(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n,
        const lbfgsfloatval_t step);
    int progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n,
        int k, int ls);

    static lbfgsfloatval_t _evaluate(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g,
        const int n, const lbfgsfloatval_t step);
    static int _progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step, int n, int k, int ls);
};

#endif /* LIBLBFGSOPTIMIZER_H_ */