/*
 * MNISTDataBinaryDigitsFunction.h
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#ifndef MNISTDATABINARYDIGITSFUNCTION_H_
#define MNISTDATABINARYDIGITSFUNCTION_H_

#include "MNISTDataFunction.h"

class MNISTDataBinaryDigitsFunction: public MNISTDataFunction
{
  public:
    void configurePolicy(const Eigen::MatrixXd& tmpX, Eigen::MatrixXd& X,
        const Eigen::MatrixXd& tmpY, Eigen::MatrixXd& Y);
};

#endif /* MNISTDATABINARYDIGITSFUNCTION_H_ */
