/*
 * MNISTSoftmaxCostFunction.h
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#ifndef SOFTMAXCOSTFUNCTION_H_
#define SOFTMAXCOSTFUNCTION_H_

#include "CostFunction.h"
#include "SoftmaxFunction.h"

class SoftmaxCostFunction: public CostFunction
{
  private:
    ActivationFunction* softmax;
    const double& LAMBDA;

  public:
    SoftmaxCostFunction(const double& LAMBDA = 0.0f);
    ~SoftmaxCostFunction();

    Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    double evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y, Eigen::VectorXd& grad);
    double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);

  private:
    Eigen::MatrixXd getMat(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);
};

#endif /* SOFTMAXCOSTFUNCTION_H_ */
