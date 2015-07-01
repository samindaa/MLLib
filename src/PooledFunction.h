/*
 * PooledFunction.h
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#ifndef POOLEDFUNCTION_H_
#define POOLEDFUNCTION_H_

#include "EigenFunction.h"

class PooledFunction: public EigenFunction
{
  public:
    Eigen::MatrixXd X;
};

#endif /* POOLEDFUNCTION_H_ */
