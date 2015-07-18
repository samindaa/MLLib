/*
 * Optimizer.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "CostFunction.h"
#include "DataFunction.h"

class Optimizer
{
  public:
    virtual ~Optimizer()
    {
    }

    virtual void optimize(Eigen::VectorXd& theta, DataFunction* dataFunction,
        CostFunction* costFunction) =0;
};

#endif /* OPTIMIZER_H_ */
