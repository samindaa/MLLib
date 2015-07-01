/*
 * LinearCostFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef LINEARCOSTFUNCTION_H_
#define LINEARCOSTFUNCTION_H_

#include "CostFunction.h"

class LinearCostFunction: public CostFunction
{
  public:
    Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    double evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y, Eigen::VectorXd& grad);
    /*Eigen::VectorXd getGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
     const Eigen::MatrixXd& Y);
     double getCost(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
     const Eigen::MatrixXd& Y);
     */
    double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);
};

#endif /* LINEARCOSTFUNCTION_H_ */
