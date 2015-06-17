/*
 * Function.h
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "Eigen/Dense"

class Function
{
  public:
    virtual ~Function()
    {
    }

    virtual Eigen::MatrixXd getFunc(const Eigen::MatrixXd& X) =0;
    virtual Eigen::MatrixXd getGrad(const Eigen::MatrixXd& FX) =0;
};

#endif /* FUNCTION_H_ */
