/*
 * LogisticCostFunction.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef MNISTCOSTFUNCTION_H_
#define MNISTCOSTFUNCTION_H_

#include "CostFunction.h"
#include "SigmoidFunction.h"

class LogisticCostFunction: public CostFunction
{
  private:
    ActivationFunction* sigmoid;

  public:
    LogisticCostFunction();
    ~LogisticCostFunction();

    Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    double evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y, Eigen::VectorXd& grad);

    double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);

};

#endif /* MNISTCOSTFUNCTION_H_ */
