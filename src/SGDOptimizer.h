/*
 * SGDOptimizer.h
 *
 *  Created on: Jul 5, 2015
 *      Author: sam
 */

#ifndef SGDOPTIMIZER_H_
#define SGDOPTIMIZER_H_

#include "Optimizer.h"

class SGDOptimizer: public Optimizer
{
  public:
    void optimize(Eigen::VectorXd& theta, DataFunction* dataFunction, CostFunction* costFunction);
};

#endif /* SGDOPTIMIZER_H_ */
