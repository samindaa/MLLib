/*
 * Pooling.h
 *
 *  Created on: Jun 29, 2015
 *      Author: sam
 */

#ifndef POOLING_H_
#define POOLING_H_

#include "EigenFunction.h"

class Pooling: public EigenFunction
{
  public:
    Eigen::MatrixXd X;
};

#endif /* POOLING_H_ */
